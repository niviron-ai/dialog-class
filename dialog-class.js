const { YdbChatMessageHistory } = require("@dialogai/ydb-chat-history");
const { ChatOpenAI } = require('@langchain/openai');
const { ChatAnthropic } = require("@langchain/anthropic");
const { RunnableWithMessageHistory, Runnable} = require("@langchain/core/runnables");
const {
    ToolMessage,
    SystemMessage,
    HumanMessage,
    BaseMessage,
    StoredMessage,
    mapChatMessagesToStoredMessages,
    mapStoredMessagesToChatMessages
} = require("@langchain/core/messages");
const { ToolNode } = require("@langchain/langgraph/prebuilt");
const { END, START, StateGraph, messagesStateReducer } = require("@langchain/langgraph");
const { convertToOpenAITool } = require("@langchain/core/utils/function_calling");
const { DynamicStructuredTool } = require("@langchain/core/tools");
const I = require('@dieugene/utils');
const db = require('@dieugene/key-value-db');
const { z } = require('zod');
const billing = require("@dieugene/billing");
const logger = require("@dieugene/logger")('DIALOG CLASS');


let default_start_system_msg = `
Ты - полезный помощник
`,
    default_format_msg = `
Ответы форматируй как простой текст. Перенос строк выполняй обычным образом. 
Какие-либо форматы разметки, в том числе HTML или Markdown, НЕ ИСПОЛЬЗУЙ.
`;

class Dialog {
    db;
    llm;
    ctx; // В случае использования с telegraf.js, доступ к объекту вызова
    tools = [];
    observers = [];
    session_messages = []; //# Обращаемся к списку сообщений в разных местах
    session_history = "";
    session_summary = undefined;
    is_over = false;
    is_started = false;
    is_restored = false;
    is_interrupted = false;
    is_tool_activated = false;
    interruption_message = 'Dialog is over';
    last_human_message;

    session_id = "";
    temp_instructions = [];
    opponent = null;
    alias = "";
    show_alias = true;
    start_system_msg = "";
    additional_starting_instructions = [];
    ignore_starting_message = false;
    storables = [];
    storage = {}; // Поле, где хранятся значения объектов, перечисленных в поле 'storables'
    restorable = ['is_over', 'is_started', 'dialog_code', 'last_human_message', 'session_summary'];
    restorable_exceptions = ['user_wants_to_speak'];
    dialog_code = '';
    tool_data = {name: 'dialog', description: 'Используется для общения с пользователем на определенную тему'};
    response = {
        status_message: '',
        message: '',
        message_buttons: [], // Массив строк "text::action",
        reply_options: [],
        post_message: undefined, /*{ text: undefined, markup_data: undefined}*/
        post_messages: [],/*{ text: undefined, markup_data: undefined}*/
        files: []  // {type: 'photo', url: ''}
    };
    utils = {
        exclude_instructions: (messages = []) => messages.filter(x => !is_instruction(x)),
        stringify_messages: (messages = []) => {
            let d = {'ai': this.alias, 'human': 'Респондент', 'system': 'System'};
            return messages.map(msg => d[msg._getType()] + ': -- ' + no_break(msg['content'])).join('\n');
        }
    };
    agent_config = {};
    summary_config = {
        threshold: 15,
        limit: 7
    };
    set_messages_dates = true;
    callbacks = [];
    logging_tags = []; // log_incoming_messages, log_summarizing_process, log_verbose
    constructor({
                    session_id,                                 /*  Идентификтор диалога (как правило - идентификатор
                                                                    пользователя)                                       */
                    alias = 'AI',                               /*  Псевдоним субъекта диалога.
                                                                    Используется в логировании диалога при тестировании
                                                                    с "оппонентом" (см.поле opponent), а также в диалоге
                                                                    с пользователем                                     */
                    dialog_code = '',                           /*  Код экземпляра диалога - используется как префикс
                                                                    при идентификации сессии                            */
                    start_system_msg = default_start_system_msg,/*  Стартовая инструкция диалога                        */
                    opponent = null,                            /*  Оппонент - другой экземпляр класса, выступающий в
                                                                    качестве собеседника.
                                                                    Используется при тестировании диалога               */
                    tool_name,                                  /*  Имя инструмента, подключаемого к ИИ-агенту для
                                                                    вызова данного диалога                              */
                    tool_description,                           /*  Описание инструмента, подключаемого к ИИ-агенту для
                                                                    вызова данного диалога. Должно начинаться со слов:
                                                                    "Используется для...".                              */
                    summary_config = {
                        threshold: 10,                          /*  Порог количества сообщений, при котором выполняется
                                                                    суммаризация                                        */
                        limit: 5                                /*  Количество сообщений, сохраняемых в истории до
                                                                    суммаризации                                        */
                    },
                    modelName = "gpt-4o",                       /*  Тип модели */
                    provider = 'openai',                        /*  Поставщик модели */
                    storables = [],                             /*  Названия пользовательских полей, хранимых в данном
                                                                    диалоге.                                            */
                    callbacks = [],                             /*  Массив callback-объектов, организованных по аналогии
                                                                    с langchain
                                                                    - on_restore_end
                                                                    - on_send_response_message_end
                                                                    - on_invoke_end                                     */
                    database
                } = {}) {
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Constructor START - session_id:', session_id, 'dialog_code:', dialog_code, 'alias:', alias);
        
        this.database = database;
        this.db = db.init('DIALOG_DATA', {database});
        this.llm = Dialog.get_llm({modelName, provider});
        this.session_id = [dialog_code, session_id].filter(s => !!s).join('::');
        this.dialog_code = dialog_code;
        this.opponent = opponent;
        this.alias = alias;
        this.start_system_msg = start_system_msg;
        this.summary_config = summary_config;
        this.storables = storables;
        this.callbacks = callbacks;
        this.agent_config = { configurable: {
                thread_id: this.session_id,
                sessionId: this.session_id
            }};
        if (tool_name) this.tool_data.name = tool_name;
        if (tool_description) this.tool_data.description = tool_description;
        
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Constructor END - final_session_id:', this.session_id, 'tools_count:', this.tools.length, 'callbacks_count:', this.callbacks.length);
    }

    log_tagged(tag = '', ...args) {
        if (this.logging_tags.includes(tag)) log(...args);
    }

    set_llm({modelName = "gpt-4o", temperature = 0, schema, provider = 'openai'} = {}) {
        this.llm = Dialog.get_llm({modelName, temperature, schema, provider});
        return this;
    }

    /**
     *
     * @param tool {DynamicStructuredTool}
     * @returns {Dialog}
     */
    add_tool(tool) {
        if (!!tool) this.tools.push(tool);
        return this;
    }

    refresh() {
        this.response = {
            status_message: '',
            message: '',
            reply_options: [],
            post_message: undefined, /*{ text: undefined, markup_data: undefined}*/
            post_messages: [],/*{ text: undefined, markup_data: undefined}*/
            files: []  // {type: 'photo', url: ''}
        };
        // this.callbacks = [];
        return this;
    }

    /**
     * Установка данных для "инструмента" (tool), на основании которого ИИ-агент инициирует данный диалог
     * @param name          Имя инструмента, подключаемого к ИИ-агенту для вызова данного диалога
     * @param description   Описание инструмента, подключаемого к ИИ-агенту для вызова данного диалога.
     *                      Должно начинаться со слов: "Используется для...".
     * @returns {Dialog}
     */
    set_tool_data(name = 'dialog', description = 'Используется для общения с пользователем на определенную тему') {
        this.tool_data = {name, description};
        return this;
    }

    /**
     * Получение инструмента для ИИ-агента, который инициирует данный диалог
     * @param supervisor {DialogSupervisor}
     * @returns {DynamicStructuredTool<ZodObjectAny>}
     */
    get_tool(supervisor) {
        let {name, description} = this.tool_data, self = this;
        return new DynamicStructuredTool({
            name, description,
            schema: z.object({
                file_id: z.string()
                    .default('')
                    .describe("Идентификаторы файлов, если они есть в сообщении пользователя, в виде списка, каждый элемент которого размещается с новой строки. Каждый элемент начинается с 'file::'. Если в сообщении пользователя идентификатора файла нет, параметр остается пустым."),
                user_message: z.string()
                    .describe("Текст сообщения от пользователя, которым инициируется и с которого начинается диалог")
            }),
            func: async ({file_id = '', user_message}) => {
                if (file_id.length > 0) user_message += '\n\nИдентификаторы файлов:\n' + file_id;
                try {
                    await supervisor.invoke_dialog(self, user_message );
                } catch (e) {
                    I.log_error(e, 'SUPERVISOR.INVOKE_DIALOG ');
                    return `Ошибка запуска работы помощника "${self.alias}". Попробуйте повторить запрос.`
                }
                return `Запрос пользователя был направлен помощнику "${self.alias}"`
            }
        })
    }

    /**
     * Метод передачи "инструмента" в Агент-супервайзер [https://github.com/Dieugene/dialog-supervisor] и
     * @param supervisor {DialogSupervisor}
     * @returns {Dialog}
     */
    get_tool_for(supervisor) {
        let tool = this.get_tool(supervisor);
        if (supervisor) {
            supervisor.reg_dialog_tool(tool);
            supervisor.reg_dialog(this);
        }
        return this;
    }

    /**
     * Установка идентификатора диалога, с учетом префикса диалога
     * @param session_id
     * @returns {Dialog}
     */
    set_session_id(session_id) {
        this.session_id = [this.dialog_code, session_id].filter(s => !!s).join('::');
        return this;
    }

    get_initial_session_id() {
        let spl = this.session_id.split('::');
        return spl.length === 2 ? spl[1] : ''
    }

    /**
     * Сохранение данных диалога
     * @returns {Promise<Dialog>}
     */
    async store() {
        await this.db.set(this.session_id, this.data());
        return this;
    }

    /**
     * Восстановление данных диалога из хранилища
     * @returns {Promise<*>}
     */
    async restore() {
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Restore START - session_id:', this.session_id, 'is_restored:', this.is_restored);
        
        if (this.is_restored) return;
        this.is_restored = true;
        let data = await this.db.get(this.session_id),
            is_restorable = key => !this.restorable_exceptions.includes(key);
            
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Restore - data from DB:', JSON.stringify(data));
        
        if (!data) return;
        this.storables.forEach(key => this.storage[key] = data[key]);
        this.restorable
            .filter(is_restorable)
            .forEach(key => {if (data[key] !== undefined) this[key] = data[key]});
        this.observers.forEach(observer => {
            for (let key in observer.data){
                if (is_restorable(key) && data[key] !== undefined)
                    observer.data[key] = data[key]
            }
        });
        I.log('DIALOG :: ', this.alias, ' :: RESTORED DATA :: ', JSON.stringify(this.data()));
        await Promise.all(
            this.callbacks
                .filter(cb => typeof cb.on_restore_end === 'function')
                .map(cb => cb.on_restore_end())
        );
        
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Restore END - restored_data:', JSON.stringify(this.data()));
        return this;
    }

    /**
     * Регистрация "наблюдателя"
     * @param observer
     * @param comment
     * @returns {Dialog}
     */
    reg_observer(observer, comment){
        this.observers.push(observer);
        observer.init_host(this, comment);
        return this;
    }

    /**
     * Чтение собственных данных, а также данных, накопленных у наблюдателей (в свойстве `data`) и их предоставление
     * в виде объекта
     */
    data(){
        let result = {};
        this.storables.forEach(key => result[key] = this.storage[key]);
        this.restorable.forEach(key => result[key] = this[key]);
        this.observers.forEach(observer => {
            for (let key in observer.data){result[key] = observer.data[key]}
        });
        return result
    }

    /**
     * Установка значение данных по заданному ключу (у себя или у наблюдателей в объекте observer.data)
     * @param key
     * @param value
     */
    set_data(key = '', value) {
        if (this.restorable.includes(key)) this[key] = value;
        else if (this.storables.includes(key)) this.storage[key] = value;
        else this.observers.forEach(observer => {
            if (observer.data.hasOwnProperty(key)) observer.data[key] = value;
        });
    }

    /**
     * Чтение истории диалога
     * @returns { YdbChatMessageHistory }
     */
    history() {
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] History called - session_id:', this.session_id);
        let history = get_session_history(this.session_id);
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] History created - constructor_name:', history.constructor.name);
        return history;
    }

    /**
     * Удаление истории диалога
     * @returns {Promise<Dialog>}
     */
    async clear_session() {
        let history = this.history();
        await history.clear();
        return this;
    }

    /**
     * Отправка тестовой информации
     * @returns {Promise<Dialog>}
     */
    async self_test() {
        let history = this.history(), messages = await history.getMessages.raw(), data = this.data();
        logger.critical('SELF TEST', {data, messages});
        return this;
    }

    /**
     * Запуск работы наблюдателей перед генерацией ответного сообщения
     * @param human_msg
     * @param messages
     * @returns {Promise<*>}
     */
    async pre_check(human_msg = "", messages = []) {
        if (this.observers.length === 0) return;
        I.log('     ::::::: Pre check :::::::\n');
        for (let i = 0; i < this.observers.length; i++){
            await this.observers[i].pre_check(human_msg, messages)
        }
        I.log('     ::::: Pre check done ::::\n');
        return this;
    }

    /**
     * Запуск работы наблюдателей после генерации ответного сообщения
     * @param ai_msg
     * @returns {Promise<boolean>}
     */
    async post_check(ai_msg = "") {
        if (this.observers.length === 0 || this.is_tool_activated) return true; // воизбежание повторного выполнения инструментов
        this.temp_instructions = [];
        I.log('     ::::::: Post check :::::::\n');
        let result = true;
        for (let i = 0; i < this.observers.length; i++){
            let is_checked = await this.observers[i].post_check(ai_msg, this.temp_instructions);
            if (!is_checked) result = false;
        }
        I.log('     ::::: Post check done ::::\n');
        return result
    }

    /**
     * Извлечение наблюдателя по имени класса
     * @param name
     * @returns {*}
     */
    get_observer(name) {
        for (let i = 0; i < this.observers.length; i++){
            if (this.observers[i].constructor.name === name) return this.observers[i]
        }
        return {}
    }

    /**
     * Проверка условий для продолжения работы диалога:
     * - true   - остановить диалог
     * - false  - не останавливать диалог
     * @param tag
     * @returns {boolean}
     */
    stop_dialog_condition(tag = ''){
        let data = this.data(), result = this.is_interrupted || (!!(data['is_finished'] && !data['user_wants_to_speak']));
        I.log('DIALOG :: STOP DIALOG CONDITION :: ', result, tag ? ' :: TAG :: ' + tag : '');
        return result
    }

    /**
     * Обновление полей класса, хранящих историю диалога в виде текста и в виде массива.
     * @param messages
     * @returns {Promise<void>}
     */
    async update_history(messages=[]) {
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Update_history START - incoming_messages_count:', messages.length);

        function filter_messages(session_messages) {
            return session_messages.filter(x => !is_instruction(x))
        }

        let history = this.history();
        
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Update_history - getting messages from history');
        let history_messages = await history.getMessages();
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Update_history - history_messages_count:', history_messages.length, 'history_messages_types:', history_messages.map(m => m._getType ? m._getType() : 'NO_GET_TYPE'));
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Update_history - history_messages_getType:', history_messages.map(m => m.getType ? m.getType() : 'NO_GET_TYPE_NO_UNDERSCORE'));

        this.session_messages = [...history_messages, ...messages];
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Update_history - session_messages_count:', this.session_messages.length, 'session_messages_types:', this.session_messages.map(m => m._getType ? m._getType() : 'NO_GET_TYPE'));
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Update_history - session_messages_getType:', this.session_messages.map(m => m.getType ? m.getType() : 'NO_GET_TYPE_NO_UNDERSCORE'));
        
        this.session_history = Dialog.stringify_messages(filter_messages(this.session_messages), {ai_alias: this.alias}).trim();
        
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Update_history END - session_history_length:', this.session_history.length);
    }

    /**
     * https://langchain-ai.github.io/langgraphjs/concepts/memory/#managing-long-conversation-history
     * @returns {Promise<void>}
     */
    async summarize_chat() {
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Summarize_chat START - session_id:', this.session_id, 'summary_config_threshold:', this.summary_config.threshold, 'summary_config_limit:', this.summary_config.limit);
        /*
        https://js.langchain.com/docs/troubleshooting/errors/INVALID_TOOL_RESULTS/
         */

        /**
         *
         * @param msg_list
         * @param threshold
         * @returns {{system_message: {}, messages_to_summarize: [], new_history: []} | boolean}
         */
        function get_summarize_structure(msg_list = [], threshold = 10) {

            function is_human(message) {
                return (message instanceof HumanMessage) || message._getType?.() === 'human' || message.type === 'human'
            }
            
            if (msg_list.length < threshold || threshold === 0) return false;
            if (threshold < 0) threshold = (-1) * threshold;
            let first_message_in_new_list_index = msg_list.length - threshold,
                first_message_in_new_list = msg_list[first_message_in_new_list_index];
            while (!is_human(first_message_in_new_list) && first_message_in_new_list_index > 5) {
                first_message_in_new_list_index--;
                first_message_in_new_list = msg_list[first_message_in_new_list_index];
                self.log_tagged('log_summarizing_process', 'first_message_in_new_list_index', first_message_in_new_list_index);
                self.log_tagged('log_summarizing_process', 'first_message_in_new_list', JSON.stringify(first_message_in_new_list));
            }
            self.log_tagged('log_summarizing_process', 'first_message_in_new_list_index FINAL', first_message_in_new_list_index);
            self.log_tagged('log_summarizing_process', 'first_message_in_new_list FINAL', JSON.stringify(first_message_in_new_list));
            return is_human(first_message_in_new_list)
                ? {
                    system_message: msg_list[0],
                    messages_to_summarize: msg_list.slice(1, first_message_in_new_list_index),
                    new_history: msg_list.slice(first_message_in_new_list_index)
                }
                : false
        }

        this.log_tagged('log_summarizing_process', 'summarize_chat INVOKED');

        let self = this,
            history = this.history(),
            limit = this.summary_config.limit;
            
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Summarize_chat - getting all messages from history');
        let all_messages = await history.getMessages();
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Summarize_chat - all_messages_count:', all_messages.length, 'all_messages_types:', all_messages.map(m => m._getType ? m._getType() : 'NO_GET_TYPE'));
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Summarize_chat - all_messages_getType:', all_messages.map(m => m.getType ? m.getType() : 'NO_GET_TYPE_NO_UNDERSCORE'));
        
        let summarize_structure = get_summarize_structure(all_messages, limit);

        if (!summarize_structure) return this.log_tagged('log_summarizing_process', 'summarize_structure is NEGATIVE');
        
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Summarize_chat - summarize_structure created, messages_to_summarize_count:', summarize_structure.messages_to_summarize.length, 'new_history_count:', summarize_structure.new_history.length);
        I.log('DIALOG :: ' + this.alias + ' :: SUMMARIZE CHAT IS INVOKED :: HISTORY ::', JSON.stringify(summarize_structure));

        let summary = this.session_summary,
            summary_message = summary
            ? `Вот резюме нашей беседы к текущему моменту:
${summary}

Расширь это резюме исходя из сообщений, представленных выше`
            : 'Сформируй резюме нашего диалога',
            chain = new ChatOpenAI({
                apiKey: process.env.OPENAI_API_KEY,
                modelName: "gpt-4o-mini",
                temperature: 0,
                configuration: {
                    basePath: process.env.PROXY_URL,
                    baseURL: process.env.PROXY_URL
                }
            });
            /*chain = get_chain_common({
                user: this.get_initial_session_id(),
                service: this.dialog_code,
                comment: this.alias + ' :: Summarize',
                chain_llm: new ChatOpenAI({
                    apiKey: process.env.OPENAI_API_KEY,
                    modelName: "gpt-4o-mini",
                    temperature: 0,
                    configuration: {
                        basePath: process.env.PROXY_URL,
                        baseURL: process.env.PROXY_URL
                    }
                })
            });*/

        let messages = [
                ...summarize_structure.messages_to_summarize,
                new HumanMessage(summary_message)
            ];
            
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Summarize_chat - invoking chain with messages_count:', messages.length, 'messages_types:', messages.map(m => m._getType ? m._getType() : 'NO_GET_TYPE'));
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Summarize_chat - invoking chain messages_getType:', messages.map(m => m.getType ? m.getType() : 'NO_GET_TYPE_NO_UNDERSCORE'));
        
        let response = await chain.invoke(messages),
            summary_add_on = `
Прими во внимание краткое описание того, о чем мы общались ранее, представленное ниже
============
${response.content}`;

        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Summarize_chat - chain response received, content_length:', response.content.length);

        this.session_summary = response.content;
        await history.clear();
        let new_history = [
            new SystemMessage(this.start_system_msg + default_format_msg + summary_add_on),
            ...summarize_structure.new_history
        ];

        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Summarize_chat - new_history prepared, count:', new_history.length, 'types:', new_history.map(m => m._getType ? m._getType() : 'NO_GET_TYPE'));
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Summarize_chat - new_history getType:', new_history.map(m => m.getType ? m.getType() : 'NO_GET_TYPE_NO_UNDERSCORE'));

        history.backup()
            .then(
                () => history.clear()
                    .then(
                        () => history.addMessages(new_history)
                            .then(() => {
                                this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Summarize_chat - new history saved to DB');
                                I.log('DIALOG :: SUMMARIZE CHAT :: ' +
                                    'NEW CHAT HISTORY IS SAVED ::', JSON.stringify(new_history));
                            })
                    )
            );
        
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Summarize_chat END - session_summary_length:', this.session_summary.length);
    }

    /**
     * Генерация ответного сообщения
     * @param human_msg
     * @param messages
     * @returns {Promise<*>}
     */
    async gen_message(human_msg, messages = []) {
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Gen_message START - human_msg_length:', human_msg.length, 'messages_count:', messages.length, 'messages_types:', messages.map(m => m._getType ? m._getType() : 'NO_GET_TYPE'));
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Gen_message START - messages_getType:', messages.map(m => m.getType ? m.getType() : 'NO_GET_TYPE_NO_UNDERSCORE'));
        
        await this.pre_check(human_msg, messages);
        if (this.stop_dialog_condition('inside')) return await this.stop(true);
        let retries = 0, result;
        do {
            retries++;
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Gen_message - retry attempt:', retries, 'temp_instructions_count:', this.temp_instructions.length);
            
            let agent = this.build(true);
            let invoke_messages = [...this.temp_instructions, ...messages];
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Gen_message - invoking agent with messages_count:', invoke_messages.length, 'messages_types:', invoke_messages.map(m => m._getType ? m._getType() : 'NO_GET_TYPE'));
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Gen_message - invoking agent with messages_getType:', invoke_messages.map(m => m.getType ? m.getType() : 'NO_GET_TYPE_NO_UNDERSCORE'));
            
            result = await agent.invoke({ messages: invoke_messages}, this.agent_config);
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Gen_message - agent result, messages_count:', result.messages ? result.messages.length : 0, 'last_message_type:', result.messages ? lastOf(result.messages)._getType() : 'NO_LAST_MESSAGE');
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Gen_message - agent result last_message_getType:', result.messages ? (lastOf(result.messages).getType ? lastOf(result.messages).getType() : 'NO_GET_TYPE_NO_UNDERSCORE') : 'NO_LAST_MESSAGE');
            // result = lastOf(result.messages)
        } while (retries < 10 && !await this.post_check(lastOf(result.messages).content));
        
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Gen_message END - final_retries:', retries, 'result_messages_count:', result.messages ? result.messages.length : 0);
        return result
    }

    /**
     * Сброс текущего статуса диалога
     * @returns {Promise<void>}
     */
    async reset() {
        //await this.clear_session();
        this.is_over = false; // непонятно, поле нигде не используется вроде.
        this.is_started = false;
        this.is_interrupted = false;
        //this.session_history = "";
        //this.session_messages = [];
    }

    build(readOnly = false) {
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Build START - readOnly:', readOnly, 'tools_count:', this.tools.length);
        
        let self = this,
            wayToContinue = state => { // -> один из вариантов: 'tool_node', 'summarize_chat', END
                self.log_tagged('log_verbose', '[DIALOG_VERBOSE] Build.wayToContinue - state_messages_count:', state.messages ? state.messages.length : 0);
                let lastMessage = lastOf(state.messages);
                self.log_tagged('log_verbose', '[DIALOG_VERBOSE] Build.wayToContinue - lastMessage_type:', lastMessage ? lastMessage._getType() : 'NO_MESSAGE');
                self.log_tagged('log_verbose', '[DIALOG_VERBOSE] Build.wayToContinue - lastMessage_getType:', lastMessage ? (lastMessage.getType ? lastMessage.getType() : 'NO_GET_TYPE_NO_UNDERSCORE') : 'NO_MESSAGE');
                
                let toolCalls = lastMessage.additional_kwargs.tool_calls;
                if (toolCalls) {
                    self.is_tool_activated = true;
                    self.log_tagged('log_verbose', '[DIALOG_VERBOSE] Build.wayToContinue - tool_calls detected, count:', toolCalls.length);
                    return 'tool_node';
                }
                self.log_tagged('log_verbose', '[DIALOG_VERBOSE] Build.wayToContinue - no tool_calls, returning END');
                return END;
            },
            // Define the function that calls the model
            callModel = async state => {
                self.log_tagged('log_verbose', '[DIALOG_VERBOSE] Build.callModel START - state_messages_count:', state.messages ? state.messages.length : 0, 'state_messages_types:', state.messages ? state.messages.map(m => m._getType ? m._getType() : 'NO_GET_TYPE') : []);
                self.log_tagged('log_verbose', '[DIALOG_VERBOSE] Build.callModel - state_messages_getType:', state.messages ? state.messages.map(m => m.getType ? m.getType() : 'NO_GET_TYPE_NO_UNDERSCORE') : []);
                
                let llm = self.llm.bindTools(self.tools.map(tool => convertToOpenAITool(tool)));
                llm.modelName = "gpt-4o";  // .bindTools возвращает не llm, а другой Runnable, поэтому modelName недоступно и нужно явно присвоить.
                //model = prompt.pipe(llm),

                let model = self.get_chain({self, llm, readOnly}),
                    response = await model.invoke(state.messages,
                        {configurable: {sessionId: this.session_id}});
                        
                self.log_tagged('log_verbose', '[DIALOG_VERBOSE] Build.callModel - response_type:', response ? response._getType() : 'NO_RESPONSE', 'response_content_length:', response ? response.content.length : 0);
                self.log_tagged('log_verbose', '[DIALOG_VERBOSE] Build.callModel - response_getType:', response ? (response.getType ? response.getType() : 'NO_GET_TYPE_NO_UNDERSCORE') : 'NO_RESPONSE');
                I.log('DIALOG :: INVOKE :: CALL MODEL :: RESPONSE', JSON.stringify(response));
                // We return a list, because this will get added to the existing list
                return { messages: [response] };
            },
            graphState = {
                messages: {
                    /* Важный момент, что эта функция и присваивает id, там где пусто, и корректно отрабатывает RemoveMessage.
                     * См.: https://github.com/langchain-ai/langgraphjs/blob/5c035de70b09be08db9858ffdfa8b4530715a9c5/libs/langgraph/src/graph/message.ts#L19
                     * Пришлось много времени потратить, чтобы понять значимость свойства 'reducer'
                     * См. https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/ (для JS даже такого раздела нет) */
                    reducer: messagesStateReducer,
                    default: () => []
                },
                summary: undefined
            };
            
        self.log_tagged('log_verbose', '[DIALOG_VERBOSE] Build - creating workflow with graph state');
        
        let workflow = new StateGraph({ channels: graphState })
                .addNode("agent", callModel)
                .addNode("tool_node", get_custom_tool_node(this.tools, new YdbChatMessageHistory({
                    sessionId: this.session_id, database: this.database, readOnly
                })))
                .addEdge(START, "agent")
                .addConditionalEdges("agent", wayToContinue)
                .addEdge("tool_node", "agent")
        ;
        
        self.log_tagged('log_verbose', '[DIALOG_VERBOSE] Build END - workflow created, compiling');
        return workflow.compile();
        // let agent = workflow.compile();
        // https://github.com/langchain-ai/langchain/discussions/21801
        // https://v03.api.js.langchain.com/classes/_langchain_core.runnables.RunnableWithMessageHistory.html#historyMessagesKey
        // https://github.com/langchain-ai/langchainjs/blob/cef9b9ef7ff4a127d129a18aa5152cb7f710f4a5/langchain-core/src/runnables/history.ts#L29
        // https://github.com/langchain-ai/langchainjs/blob/cef9b9ef7ff4a127d129a18aa5152cb7f710f4a5/langchain-core/src/runnables/history.ts#L103
    }

    get_chain({self, llm, readOnly = false} = {}) {
        self = self || this;
        llm = llm || self.llm;
        
        self.log_tagged('log_verbose', '[DIALOG_VERBOSE] Get_chain - session_id:', self.session_id, 'readOnly:', readOnly, 'llm_model:', llm.modelName);
        
        return new RunnableWithMessageHistory({
            /*runnable: billing.llm.apply_usage(llm, {
                user: self.get_initial_session_id(),
                service: self.dialog_code,
                comment: self.alias
            }),*/
            runnable: llm,
            getMessageHistory: (sessionId) => {
                self.log_tagged('log_verbose', '[DIALOG_VERBOSE] Get_chain.getMessageHistory - sessionId:', sessionId, 'readOnly:', readOnly);
                return new YdbChatMessageHistory({ sessionId, readOnly, database: self.database });
            }
        });
    }

    async invoke(human_msg = "Привет") {
        if (I.notEmpty(human_msg) && this.set_messages_dates) {
            let dt = I.getDateISO(null);
            human_msg += "\n\n[Meta-Data]\nDate: " + dt.date + "\nTime: " + dt.time + "\n[/Meta-Data]";
        }
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke START - human_msg_length:', human_msg.length, 'session_id:', this.session_id);
        this.log_tagged('log_incoming_messages', 'DIALOG IS INVOKED :: WITH :: ', human_msg);
        // I.log('DIALOG IS INVOKED :: WITH :: ', human_msg);
        
        await this.restore();
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - after restore, is_restored:', this.is_restored);
        
        let result, history = this.history(), self = this, function_scenario = '';
        this.last_human_message = human_msg;
        
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - getting session messages from history');
        this.session_messages = await history.getMessages();
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - session_messages_count:', this.session_messages.length, 'session_messages_types:', this.session_messages.map(m => m._getType ? m._getType() : 'NO_GET_TYPE'));
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - session_messages_getType:', this.session_messages.map(m => m.getType ? m.getType() : 'NO_GET_TYPE_NO_UNDERSCORE'));
        
        if (this.session_messages.length > 0) {
            function_scenario = 'this.session_messages.length > 0';
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - existing session scenario, checking stop condition');
            
            if (this.stop_dialog_condition('before')) return await this.stop(); //!!!!!!!!!!!!!!!!!!!!!!

            let msg = new HumanMessage(human_msg), messages = [ msg ];
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - created HumanMessage, type:', msg._getType(), 'content_length:', msg.content.length);
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - created HumanMessage getType:', msg.getType ? msg.getType() : 'NO_GET_TYPE_NO_UNDERSCORE');
            
            this.session_messages.push(msg);
            this.session_history = this.utils.stringify_messages(
                this.utils.exclude_instructions(this.session_messages)
            ).trim();

            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - before gen_message, messages_count:', messages.length, 'messages_types:', messages.map(m => m._getType ? m._getType() : 'NO_GET_TYPE'));
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - before gen_message, messages_getType:', messages.map(m => m.getType ? m.getType() : 'NO_GET_TYPE_NO_UNDERSCORE'));
            result = await this.gen_message(human_msg, messages);
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - after gen_message, result_messages_count:', result.messages ? result.messages.length : 0);

            if (this.stop_dialog_condition('after')) return await this.stop(true);
            I.log('DIALOG MESSAGES & RESULT');
            I.log('MESSAGES ::', JSON.stringify(messages));
            I.log('RESULT ::', JSON.stringify(result));
            // await history.addMessages([...messages, lastOf(result.messages)]);
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - adding result messages to history, count:', result.messages ? result.messages.length : 0);
            await history.addMessages([...result.messages]);
        }
        else {
            function_scenario = 'this.session_messages.length === 0';
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - new session scenario, creating initial messages');
            
            let initials = [this.start_system_msg, ...this.additional_starting_instructions, default_format_msg].join('\n\n'),
                messages = [new SystemMessage(initials)];
                
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - created SystemMessage, type:', messages[0]._getType(), 'content_length:', messages[0].content.length);
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - created SystemMessage getType:', messages[0].getType ? messages[0].getType() : 'NO_GET_TYPE_NO_UNDERSCORE');
            
            if (!this.ignore_starting_message && human_msg !== '')
                messages = [...messages, new HumanMessage(human_msg)];
                
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - initial messages prepared, count:', messages.length, 'types:', messages.map(m => m._getType()));
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - initial messages getType:', messages.map(m => m.getType ? m.getType() : 'NO_GET_TYPE_NO_UNDERSCORE'));
            
            let agent = this.build();
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - agent built, invoking with messages', JSON.stringify(messages));
            
            result = await agent.invoke({ messages }, this.agent_config);
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - agent invoked, result_messages_count:', result.messages ? result.messages.length : 0);
        }
        
        result = lastOf(result.messages);
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - final result extracted, type:', result ? result._getType() : 'NO_RESULT', 'content_length:', result ? result.content.length : 0);
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke - final result getType:', result ? (result.getType ? result.getType() : 'NO_GET_TYPE_NO_UNDERSCORE') : 'NO_RESULT');
        
        if (!result) await logger.critical('WARNING :: ERROR at SUPERVISOR.INVOKE_DIALOG :: INFO', {function_scenario});
        let response = result.content;
        I.log(break_lines(this.alias + ": -- " + response) + '\n\n');
        await this.store();
        this.response.message = response;
        if (this.opponent !== null) {
            await timeout(1000);
            this.opponent.invoke(response)
        }

        await Promise.all(
            this.callbacks
                .filter(cb => typeof cb.on_invoke_end === 'function')
                .map(cb => cb.on_invoke_end())
        );
        
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Invoke END - response_length:', response.length, 'function_scenario:', function_scenario);
        return response
    }

    async invoke_with_instruction(instruction_text = '') {
        return await this.invoke(Dialog.get_instruction.as_human_text(instruction_text))
    }

    async check_for_summary_invocation() {
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Check_for_summary_invocation START - session_messages_count:', this.session_messages.length, 'threshold:', this.summary_config.threshold);
        this.log_tagged('log_summarizing_process', 'check_for_summary_invocation INVOKED');
        
        if (this.session_messages.length < this.summary_config.threshold) {
            this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Check_for_summary_invocation - threshold not reached, returning false');
            return false;
        }
        
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Check_for_summary_invocation - threshold reached, calling summarize_chat');
        await this.summarize_chat();
        
        this.log_tagged('log_verbose', '[DIALOG_VERBOSE] Check_for_summary_invocation END - returning true');
        return true
    }

    /**
     * Метод остановки диалога с возвратом сообщения, записанного в поле this.interruption_message
     * @param save_data
     * @returns {Promise<string>}
     */
    async stop(save_data = false) {
        this.is_over = true; // непонятно, поле нигде не используется вроде.
        if (save_data) await this.store();
        return this.interruption_message;
    }

    /**
     * Метод предоставления данных по экземпляру класса в виде строки
     * @returns {string}
     */
    toString() {
        let p = {};
        p['session_id'] = this.session_id;
        p['is_over'] = this.is_over;
        p['opponent'] = this.opponent !== null ? this.opponent.alias : 'None';
        p['session_history'] = this.session_history;
        p['session_messages'] = this.session_messages;
        p['data'] = this.data;
        return JSON.stringify(p)
    }

    /**
     * Удаляет блоки метаданных из текста ответа ИИ
     * Блоки начинаются с [Meta-Data] и заканчиваются [/Meta-Data]
     * @param {string} text - Исходный текст с возможными блоками метаданных
     * @returns {string} - Текст без блоков метаданных
     */
    remove_metadata_blocks(text) {
        if (typeof text !== 'string' || I.env_log_has('SHOW_DIALOG_METADATA')) return text;

        // Удаляем все блоки [Meta-Data]...[/Meta-Data] из текста
        // [\s\S]*? - любые символы включая переносы строк, минимальное совпадение
        return text.replace(/\[Meta-Data\][\s\S]*?\[\/Meta-Data\]/g, '').trim();
    }

    /**
     * Статический метод для обращения к LLM (используется для упрощения - без необходимости импорта и инициализации
     * соответствующих LangChain-классов.
     * @param message
     * @param modelName
     * @param temperature
     * @param systemPrompt
     * @returns {Promise<MessageContent>}
     */
    static async call_llm(message, {modelName = "gpt-4o", temperature = 0, systemPrompt = 'Ты - полезный помощник.', schema, provider = 'openai'} = {}) {
        I.log('DIALOG CLASS :: CALLING LLM :: ', message);
        let llm = this.get_llm({modelName, temperature, provider, schema}),
        response = await llm.invoke(
            [new SystemMessage(systemPrompt), new HumanMessage(message)]
        );
        return !!schema ? response : response.content;
    }

    static get_llm({modelName = "gpt-4o", temperature = 0, schema, provider = 'openai'} = {}) {
        let llm = provider ===  'openai' ? new ChatOpenAI({
            apiKey: process.env.OPENAI_API_KEY,
            modelName,
            temperature,
            configuration: {
                basePath: process.env.PROXY_URL,
                baseURL: process.env.PROXY_URL
            }
        }) : new ChatAnthropic({
            // apiKey: process.env.OPENAI_API_KEY,
            modelName: modelName,
            temperature: 0
        });
        return !!schema ? llm.withStructuredOutput(schema) : llm;
    }

    /**
     * Статический метод получения дополнительных инструкций для LLM. Учитывая, что некоторые LLM (как, например,
     * Claude), принимают системные сообщения только в начале, способ генерации инструкций разнится.
     * @param instruction
     * @param model
     * @param as_human
     * @returns {any}
     */
    static get_instruction(instruction="", {model = 'OpenAI', as_human = false} = {}) {

        return model ==='OpenAI' && !as_human
            ? new SystemMessage(instruction)
            : new HumanMessage(instruction_message_wrap.format(instruction))
    }

    /**
     * Генерация инструкции и добавление ее в начало массива сообщений.
     * @param messages
     * @param instruction
     */
    static add_instruction(messages=[], instruction="") {
        messages.unshift(Dialog.get_instruction(instruction))
    }

    /**
     * Проверка, является ли сообщение инструкцией. Используется для фильтрации инструкций при текстовом отображении
     * истории диалога.
     * @param msg
     * @returns {boolean|ZodString|*}
     */
    static is_instruction(msg, mindTools = true) {
        return is_instruction(msg, mindTools)
    }

    static async get_data(id, field_name, database) {
        let kv_db = db.init('DIALOG_DATA', { database }),
            data_list = await kv_db.get.where_id_ends_with(id);
        if (!field_name) return data_list;
        let value;
        while (!value && data_list.length > 0) {
            value = data_list.pop().data[field_name];
        }
        I.log('DIALOG :: STATIC :: GET DATA :: ID :: ', id, ' :: FIELD NAME :: ', field_name, ' :: VALUE :: ', value);
        return value;
    }

    static stringify_messages(messages = [], {ai_alias = 'ai', user_alias = 'Респондент', system_alias = 'System', tool_alias = 'tool'} = {}){
        log('[DIALOG_VERBOSE] Static.stringify_messages - input_messages_count:', messages.length, 'input_messages_types:', messages.map(m => m._getType ? m._getType() : 'NO_GET_TYPE'));
        log('[DIALOG_VERBOSE] Static.stringify_messages - input_messages_getType:', messages.map(m => m.getType ? m.getType() : 'NO_GET_TYPE_NO_UNDERSCORE'));
        let d = {'ai': ai_alias, 'human': user_alias, 'system': system_alias, 'tool': tool_alias};
        let result = messages.map(msg => d[msg._getType()] + ': -- ' + no_break(msg['content'])).join('\n');
        log('[DIALOG_VERBOSE] Static.stringify_messages - result_length:', result.length);
        return result;
    }

    static serialize_messages(messages = []) {
        log('[DIALOG_VERBOSE] Static.serialize_messages - input_messages_count:', messages.length, 'input_messages_types:', messages.map(m => m._getType ? m._getType() : 'NO_GET_TYPE'));
        log('[DIALOG_VERBOSE] Static.serialize_messages - input_messages_getType:', messages.map(m => m.getType ? m.getType() : 'NO_GET_TYPE_NO_UNDERSCORE'));
        let result = mapChatMessagesToStoredMessages(messages);
        log('[DIALOG_VERBOSE] Static.serialize_messages - result_count:', result.length);
        return result;
    }

    static deserialize_messages(messages = []) {
        log('[DIALOG_VERBOSE] Static.deserialize_messages - input_messages_count:', messages.length);
        let result = mapStoredMessagesToChatMessages(messages);
        log('[DIALOG_VERBOSE] Static.deserialize_messages - result_count:', result.length, 'result_types:', result.map(m => m._getType ? m._getType() : 'NO_GET_TYPE'));
        log('[DIALOG_VERBOSE] Static.deserialize_messages - result_getType:', result.map(m => m.getType ? m.getType() : 'NO_GET_TYPE_NO_UNDERSCORE'));
        return result;
    }
}

Dialog.get_instruction.as_human_text = function (instruction="") {
    return instruction_message_wrap.format(instruction)
};

let instruction_message_wrap = `
Инструкция: {0}.
Данное сообщение не комментируй.
`;

/**
 *
 * @param user
 * @param service
 * @param comment
 * @param chain_llm
 * @returns {*}
 */
function get_chain_common({user, service, comment, chain_llm} = {}) {
    I.log('DIALOG :: GET COMMON CHAIN :: LLM :: MODEL NAME :: ', chain_llm.modelName);
    return billing.llm.apply_usage(chain_llm, {user, service, comment});
}
/**
 *
 * @param sessionId
 * @returns {YdbChatMessageHistory}
 */
function get_session_history(sessionId) {
    log('[DIALOG_VERBOSE] get_session_history - sessionId:', sessionId);
    let history = new YdbChatMessageHistory({sessionId});
    log('[DIALOG_VERBOSE] get_session_history - created history instance, constructor:', history.constructor.name);
    return history;
}

function no_break(s = "") {
    return s.startsWith('\n') ?  s.slice(1) : s
}

function break_lines(lines) {
    function break_single_line(txt) {
        function chunks(arr, size) {
            let result = [], buff = '';
            arr.forEach(s => {
                if ((s.length + buff.length) <= size) buff = buff + ' ' + s;
                else {
                    result.push(buff);
                    buff = s;
                }
            });
            if (buff.length > 0) result.push(buff);
            return result;
        }

        if (txt === "") return txt;
        let txt_list = txt.split(' '),
            txt_strings = chunks(txt_list, 100);
        return txt_strings.join('\n')
    }

    return lines.split('\n').map(s => break_single_line(s)).join('\n')
}

function timeout(time = 150) {
    return new Promise(resolve => setTimeout(resolve, time));
}

function is_instruction(msg, mindTools = true) {
    if (typeof msg === 'string') return is_instruction_text(msg);
    let type = msg._getType(), text = no_break(msg.content);
    let result = type === 'system' || (mindTools && type === 'tool') || is_instruction_text(text);
    return result;
}

/**
 * Упрощенная функция проверки инструкций
 * @param {string} text - Текст сообщения
 * @returns {boolean} True если это инструкция
 */
function is_instruction_text(text) {
    if (!text || typeof text !== 'string') return false;
    let nb = no_break(text);
    const instructionPatterns = [
        /^\[Meta-Data\]/i,
        /^Uncertainty Level:/i,
        /^What I Know:/i,
        /^What I Need:/i,
        /^Phase:/i,
        /^Action:/i,
        /^\[\/Meta-Data\]/i,
        /^ТЫ ВЕДЕШЬ ДВУХФАЗНЫЙ ДИАЛОГ/i,
        /^ФАЗА \d+:/i,
        /^ОБЯЗАТЕЛЬНО В КАЖДОМ ОТВЕТЕ:/i,
        /^УРОВНИ НЕОПРЕДЕЛЕННОСТИ:/i,
        /^КРИТЕРИИ ДЛЯ UNCERTAINTY LEVEL:/i,
        /^ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:/i,
        /^ВАЖНО:/i
    ];
    return nb.startsWith('Инструкция: ') || instructionPatterns.some(pattern => pattern.test(text));
}

function lastOf(arr = []) {return arr[arr.length - 1]}

function log(...args) {
    I.log_if('DIALOG', ...args)
}

/**
 * @see https://langchain-ai.github.io/langgraphjs/how-tos/tool-calling-errors/#custom-strategies
 * @param tools
 * @param history {YdbChatMessageHistory}
 * @returns {function(*): {messages: Array}}
 */
function get_custom_tool_node(tools = [], history) {
    let toolsByName = {};
    tools.forEach(tool => toolsByName[tool.name] = tool);
    return async (state) => {
        const { messages } = state;
        const lastMessage = messages[messages.length - 1];
        const outputMessages = [];
        
        // Добавляем логирование в начале функции
        log('[DIALOG_VERBOSE] get_custom_tool_node START - messages_count:', messages ? messages.length : 0, 'lastMessage_type:', lastMessage ? lastMessage._getType() : 'NO_LAST_MESSAGE', 'tool_calls_count:', lastMessage ? lastMessage.tool_calls.length : 0);
        log('[DIALOG_VERBOSE] get_custom_tool_node START - lastMessage_getType:', lastMessage ? (lastMessage.getType ? lastMessage.getType() : 'NO_GET_TYPE_NO_UNDERSCORE') : 'NO_LAST_MESSAGE');
        
        if (lastMessage.tool_calls.length > 1)
            I.log('CUSTOM TOOL NODE :: GOT', lastMessage.tool_calls.length, 'TOOLS ::', JSON.stringify(lastMessage.tool_calls));
        for (const toolCall of lastMessage.tool_calls) {
            try {
                I.log('CUSTOM TOOL NODE IS INVOKED :: BEFORE :: ', JSON.stringify(toolCall));
                log('[DIALOG_VERBOSE] get_custom_tool_node - invoking tool:', toolCall.name, 'tool_call_id:', toolCall.id);
                
                const toolResult = await toolsByName[toolCall.name].invoke(toolCall);
                log('[DIALOG_VERBOSE] get_custom_tool_node - tool result type:', toolResult ? toolResult._getType() : 'NO_GET_TYPE', 'result_length:', toolResult ? toolResult.content.length : 0);
                log('[DIALOG_VERBOSE] get_custom_tool_node - tool result getType:', toolResult ? (toolResult.getType ? toolResult.getType() : 'NO_GET_TYPE_NO_UNDERSCORE') : 'NO_TOOL_RESULT');
                
                I.log('CUSTOM TOOL NODE IS INVOKED :: AFTER :: ', JSON.stringify(toolResult));
                outputMessages.push(toolResult);
            } catch (error) {
                // Return the error if the tool call fails
                I.log('CUSTOM TOOL NODE ERROR :: Tool name ::', toolCall.name, ':: MESSAGE ::', error.message);
                I.log('CUSTOM TOOL NODE ERROR :: Tool name ::', toolCall.name, ':: STACK ::', error.stack.replaceAll('\n', ' '));
                
                log('[DIALOG_VERBOSE] get_custom_tool_node - tool error:', toolCall.name, 'error_message:', error.message);
                
                let errorMessage = new ToolMessage({
                    content: error.message,
                    name: toolCall.name,
                    tool_call_id: toolCall.id,
                    additional_kwargs: { error }
                });
                
                log('[DIALOG_VERBOSE] get_custom_tool_node - created error ToolMessage, type:', errorMessage._getType());
                log('[DIALOG_VERBOSE] get_custom_tool_node - created error ToolMessage getType:', errorMessage.getType ? errorMessage.getType() : 'NO_GET_TYPE_NO_UNDERSCORE');
                outputMessages.push(errorMessage);
            }
        }
        
        log('[DIALOG_VERBOSE] get_custom_tool_node - outputMessages_count:', outputMessages.length, 'outputMessages_types:', outputMessages.map(m => m._getType ? m._getType() : 'NO_GET_TYPE'));
        log('[DIALOG_VERBOSE] get_custom_tool_node - outputMessages_getType:', outputMessages.map(m => m.getType ? m.getType() : 'NO_GET_TYPE_NO_UNDERSCORE'));
        
        // Ради чего весь сыр-бор... Записать результат в историю, чтобы не возникала ошибка
        // https://js.langchain.com/docs/troubleshooting/errors/INVALID_TOOL_RESULTS/
        log('[DIALOG_VERBOSE] get_custom_tool_node - adding messages to history, count:', outputMessages.length);
        await history.addMessages(outputMessages);
        log('[DIALOG_VERBOSE] get_custom_tool_node END - returning messages_count:', outputMessages.length);
        
        return { messages: outputMessages };
    };
}

module.exports = {Dialog};