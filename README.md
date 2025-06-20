# @dialogai/dialog-class

Мощный и гибкий класс для создания и управления AI-диалогами с поддержкой различных языковых моделей, инструментов и сохранения состояния.

## Возможности

- 🤖 **Поддержка множественных провайдеров**: OpenAI GPT и Anthropic Claude
- 🛠️ **Система инструментов**: Интеграция с LangChain tools для расширения функциональности
- 💾 **Персистентность**: Автоматическое сохранение и восстановление состояния диалога
- 📝 **Управление историей**: Умная система суммаризации длинных диалогов
- 🔄 **Callbacks**: Система обратных вызовов для мониторинга и кастомизации
- 🎯 **Инструменты агентов**: Возможность использовать диалог как инструмент для других AI-агентов
- 💰 **Биллинг**: Интегрированный мониторинг использования и расходов

## Установка

```bash
npm install @dialogai/dialog-class
```

## Зависимости

Модуль использует следующие основные зависимости:
- `@langchain/openai` - для работы с OpenAI GPT моделями
- `@langchain/anthropic` - для работы с Anthropic Claude моделями
- `@dialogai/ydb-chat-history` - для сохранения истории диалогов
- `@dieugene/billing` - для мониторинга использования
- `@dieugene/key-value-db` - для хранения данных
- `@dieugene/utils` - утилиты

## Основные возможности

### Инициализация диалога

```javascript
const { Dialog } = require('@dialogai/dialog-class');

const dialog = new Dialog({
    session_id: 'user123',
    alias: 'Помощник',
    dialog_code: 'support',
    start_system_msg: 'Ты - дружелюбный помощник службы поддержки',
    modelName: 'gpt-4o',
    provider: 'openai',
    database: 'your_database_connection_string', // Опционально, если не указано - используется YDB_ADDRESS
    summary_config: {
        threshold: 15,
        limit: 7
    }
});
```

### Параметры конструктора

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `session_id` | string | - | Идентификатор диалога (обычно ID пользователя) |
| `alias` | string | 'AI' | Псевдоним AI-ассистента |
| `dialog_code` | string | '' | Код экземпляра диалога (префикс для session_id) |
| `start_system_msg` | string | - | Начальная системная инструкция |
| `opponent` | Dialog | null | Другой экземпляр для тестирования диалогов |
| `tool_name` | string | - | Имя инструмента для AI-агентов |
| `tool_description` | string | - | Описание инструмента |
| `summary_config` | object | `{threshold: 10, limit: 5}` | Настройки суммаризации |
| `modelName` | string | 'gpt-4o' | Модель языка |
| `provider` | string | 'openai' | Провайдер ('openai' или 'anthropic') |
| `storables` | array | [] | Поля для сохранения в базе данных |
| `callbacks` | array | [] | Массив callback-функций |
| `database` | string | - | Строка подключения к базе данных для хранения истории и данных диалога |

### Проведение диалога

```javascript
// Отправка сообщения
const response = await dialog.invoke("Привет! Как дела?");
console.log(response.message);

// Отправка с дополнительной инструкцией
await dialog.invoke_with_instruction("Отвечай кратко и по делу");
```

### Добавление инструментов

```javascript
const { DynamicStructuredTool } = require("@langchain/core/tools");
const { z } = require('zod');

const weatherTool = new DynamicStructuredTool({
    name: "get_weather",
    description: "Получить информацию о погоде в указанном городе",
    schema: z.object({
        city: z.string().describe("Название города")
    }),
    func: async ({city}) => {
        // Логика получения погоды
        return `Погода в ${city}: солнечно, +25°C`;
    }
});

dialog.add_tool(weatherTool);
```

### Управление состоянием

```javascript
// Сохранение состояния
await dialog.store();

// Восстановление состояния
await dialog.restore();

// Сохранение пользовательских данных
dialog.set_data('user_preference', 'краткие ответы');

// Получение данных
const preference = dialog.data().user_preference;
```

### Система callbacks

```javascript
const dialog = new Dialog({
    session_id: 'user123',
    callbacks: [
        {
            on_invoke_end: (result) => {
                console.log('Диалог завершен:', result);
            },
            on_restore_end: () => {
                console.log('Состояние восстановлено');
            }
        }
    ]
});
```

### Использование как инструмент

```javascript
// Настройка диалога как инструмента для другого агента
dialog.set_tool_data(
    'customer_support', 
    'Используется для общения с клиентами службы поддержки'
);

const tool = dialog.get_tool(supervisor);
```

## Продвинутые возможности

### Суммаризация диалогов

Диалог автоматически суммаризирует длинные беседы для оптимизации использования токенов:

```javascript
const dialog = new Dialog({
    session_id: 'user123',
    summary_config: {
        threshold: 20, // Суммаризировать после 20 сообщений
        limit: 8       // Оставлять последние 8 сообщений
    }
});
```

### Наблюдатели (Observers)

```javascript
const observer = {
    name: 'logger',
    on_message: (message) => console.log('Новое сообщение:', message),
    on_error: (error) => console.error('Ошибка:', error)
};

dialog.reg_observer(observer, 'Логгер сообщений');
```

### Статические методы

```javascript
// Прямой вызов LLM без диалога
const response = await Dialog.call_llm(
    "Объясни квантовую физику",
    {
        modelName: "gpt-4o",
        temperature: 0.7,
        systemPrompt: "Ты - учитель физики"
    }
);

// Получение экземпляра LLM
const llm = Dialog.get_llm({
    modelName: "claude-3-sonnet-20240229",
    provider: "anthropic"
});

// Работа с сообщениями
const instruction = Dialog.get_instruction("Будь вежливым");
Dialog.add_instruction(messages, "Отвечай кратко");
```

## Структура ответа

Объект `dialog.response` содержит:

```javascript
{
    status_message: '',     // Статусное сообщение
    message: '',           // Основной текст ответа
    reply_options: [],     // Варианты быстрых ответов
    post_message: undefined, // Дополнительное сообщение
    post_messages: [],     // Массив дополнительных сообщений
    files: []             // Файлы для отправки
}
```

## Переменные окружения

```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PROXY_URL=https://your-proxy-url.com  # Опционально
YDB_ADDRESS=your_ydb_database_url     # Используется по умолчанию, если параметр database не указан
```

## Методы жизненного цикла

```javascript
// Проверки перед и после обработки
dialog.pre_check = async (humanMsg, messages) => {
    // Логика предварительной проверки
    return true;
};

dialog.post_check = async (aiMsg) => {
    // Логика постобработки
    return aiMsg;
};

// Условие остановки диалога
dialog.stop_dialog_condition = (tag) => {
    return tag === 'END_CONVERSATION';
};
```

## Логирование

```javascript
// Включение логирования для отладки
dialog.logging_tags = ['log_incoming_messages', 'log_summarizing_process'];
```

## Тестирование

```javascript
// Автоматический тест диалога
const testResult = await dialog.self_test();
console.log('Результат теста:', testResult);

// Тестирование с оппонентом
const opponent = new Dialog({session_id: 'opponent'});
dialog.opponent = opponent;
```

## Примеры использования

### Простой чат-бот

```javascript
const dialog = new Dialog({
    session_id: 'chatbot_user',
    start_system_msg: 'Ты - дружелюбный чат-бот. Отвечай коротко и по существу.'
});

const response = await dialog.invoke("Расскажи анекдот");
console.log(response.message);
```

### Служба поддержки с инструментами

```javascript
const supportDialog = new Dialog({
    session_id: 'support_ticket_123',
    alias: 'Агент поддержки',
    start_system_msg: 'Ты - агент службы поддержки. Помогай клиентам решать их проблемы.'
});

// Добавление инструментов для работы с базой знаний, билетами и т.д.
supportDialog.add_tool(knowledgeBaseTool);
supportDialog.add_tool(ticketManagementTool);

await supportDialog.invoke("У меня проблема с заказом №12345");
```

## Лицензия

ISC

## Автор

Eugene Ditkovsky
