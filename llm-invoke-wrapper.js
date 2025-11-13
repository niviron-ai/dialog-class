const I = require('@dieugene/utils');

/**
 * Обертка для безопасного вызова LLM с retry и сбором диагностической информации
 * При ошибке выбрасывает детальную информацию для разработчиков
 * 
 * @param {Object} context - Контекст диалога (this из Dialog класса)
 * @param {Function} invokeFunction - функция для вызова LLM
 * @param {*} params - параметры для вызова
 * @param {Object} options - опции retry
 * @param {number} options.maxRetries - максимальное количество попыток (по умолчанию 3)
 * @param {number} options.initialDelay - начальная задержка между попытками в мс (по умолчанию 1000)
 * @param {number} options.backoffMultiplier - множитель увеличения задержки (по умолчанию 2)
 * @returns {Promise<*>}
 */
async function safe_llm_invoke(context, invokeFunction, params, options = {}) {
    const {
        maxRetries = 3,
        initialDelay = 1000,
        backoffMultiplier = 2
    } = options;

    const attempts = [];
    let delay = initialDelay;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        const attemptStartTime = Date.now();
        
        let llmLoggingPayload = {};
        try {
            const chain = params?.chain ?? null;
            const candidateLLM =
                chain?.llm ??
                chain?.bound?.llm ??
                params?.llm ??
                null;
            llmLoggingPayload = {
                chainType: chain?.constructor?.name ?? null,
                llmType: candidateLLM?.constructor?.name ?? null,
                llmModel:
                    candidateLLM?.modelName ??
                    candidateLLM?.model ??
                    candidateLLM?.lc_kwargs?.model ??
                    candidateLLM?.lc_kwargs?.modelName ??
                    null
            };
        } catch (introspectionError) {
            llmLoggingPayload = { llmIntrospectionError: introspectionError.message };
        }

        console.info('[SAFE_LLM_INVOKE] Attempt start', JSON.stringify({
            attempt,
            maxRetries,
            sessionId: context?.session_id ?? null,
            dialogCode: context?.dialog_code ?? null,
            alias: context?.alias ?? null,
            messagesCount: params?.messages?.length || 0,
            ...llmLoggingPayload
        }));

        try {
            const response = await invokeFunction(params);
            const duration = Date.now() - attemptStartTime;
            
            // Успех - возвращаем результат
            return response;
            
        } catch (error) {
            const duration = Date.now() - attemptStartTime;
            
            // Измеряем объем данных в сообщениях
            const messagesSize = calculateMessagesSize(params.messages || []);
            
            // Собираем информацию о попытке
            const attemptInfo = {
                attempt,
                durationMs: duration,
                timestamp: new Date(attemptStartTime).toISOString(),
                messagesCount: params.messages?.length || 0,
                messagesSize: messagesSize,
                errorType: error.constructor.name,
                errorMessage: error.message,
                errorCode: error.code,
                errorStatus: error.status,
                // Анализ типа ошибки
                isTimeout: isTimeoutError(error),
                isNetwork: isNetworkError(error),
                isConnection: isConnectionError(error),
                // Дополнительная информация из ошибки
                responseStatus: error.response?.status,
                responseStatusText: error.response?.statusText,
                responseData: error.response?.data ? 
                    (typeof error.response.data === 'string' ? 
                        error.response.data.substring(0, 200) : 
                        JSON.stringify(error.response.data).substring(0, 200)
                    ) : undefined,
                // Сокращенный стек (первые 5 строк)
                stackPreview: error.stack?.split('\n').slice(0, 5).join('\n')
            };
            
            attempts.push(attemptInfo);
            
            // Если это последняя попытка - выбрасываем детальную ошибку
            if (attempt === maxRetries) {
                const totalDuration = attempts.reduce((sum, a) => sum + a.durationMs, 0);
                
                const detailedError = new Error(
                    `❌ LLM INVOKE FAILED after ${maxRetries} attempts\n` +
                    `Session: ${context.session_id}\n` +
                    `Dialog: ${context.dialog_code} (${context.alias})\n` +
                    `Total duration: ${totalDuration}ms\n` +
                    `Messages: ${messagesSize.totalChars} chars (${messagesSize.humanReadable}), ${params.messages?.length || 0} messages\n` +
                    `\n📊 DIAGNOSTIC INFO (send to developers): ` +
                    JSON.stringify({
                        session_id: context.session_id,
                        dialog_code: context.dialog_code,
                        alias: context.alias,
                        totalDurationMs: totalDuration,
                        messagesStats: messagesSize,
                        attempts: attempts,
                        lastError: {
                            message: error.message,
                            code: error.code,
                            status: error.status,
                            stack: error.stack
                        },
                        environment: {
                            proxyUrl: process.env.PROXY_URL ? 'configured' : 'not configured',
                            nodeVersion: process.version,
                            platform: process.platform
                        }
                    })
                );
                
                detailedError.diagnosticInfo = { 
                    attempts, 
                    originalError: error,
                    messagesStats: messagesSize
                };
                
                // Выводим в консоль детальную информацию
                console.error('\n' + '='.repeat(80));
                console.error(detailedError.message);
                console.error('='.repeat(80) + '\n');
                
                throw detailedError;
            }
            
            // Ждем перед следующей попыткой
            console.warn(`⚠️ LLM invoke attempt ${attempt} failed (${duration}ms): ${error.message}. Retrying in ${delay}ms...`);
            await I.timeout(delay);
            delay = Math.min(delay * backoffMultiplier, 10000);
        }
    }
}

/**
 * Подсчет объема данных в сообщениях
 * @param {Array} messages - массив сообщений
 * @returns {Object} - статистика по размеру
 */
function calculateMessagesSize(messages = []) {
    let totalChars = 0;
    const breakdown = {};
    
    messages.forEach(msg => {
        try {
            const type = msg._getType ? msg._getType() : (msg.type || 'unknown');
            const content = msg.content || '';
            const contentLength = typeof content === 'string' ? 
                content.length : 
                JSON.stringify(content).length;
            
            totalChars += contentLength;
            breakdown[type] = (breakdown[type] || 0) + contentLength;
        } catch (e) {
            // Игнорируем ошибки при подсчете
        }
    });
    
    return {
        totalChars,
        humanReadable: formatBytes(totalChars),
        messagesCount: messages.length,
        breakdown
    };
}

/**
 * Форматирование размера в человекочитаемый формат
 * @param {number} chars - количество символов
 * @returns {string}
 */
function formatBytes(chars) {
    if (chars < 1024) return chars + ' chars';
    if (chars < 1024 * 1024) return (chars / 1024).toFixed(2) + ' KB';
    return (chars / (1024 * 1024)).toFixed(2) + ' MB';
}

/**
 * Определение timeout ошибки
 */
function isTimeoutError(error) {
    return error.code === 'ETIMEDOUT' || 
           error.code === 'ESOCKETTIMEDOUT' ||
           error.message?.toLowerCase().includes('timeout') ||
           error.name === 'TimeoutError';
}

/**
 * Определение сетевой ошибки
 */
function isNetworkError(error) {
    return error.code === 'ECONNREFUSED' ||
           error.code === 'ENOTFOUND' ||
           error.code === 'ECONNRESET' ||
           error.code === 'EPIPE' ||
           error.message?.toLowerCase().includes('network');
}

/**
 * Определение ошибки соединения
 */
function isConnectionError(error) {
    return error.message?.toLowerCase().includes('connection') ||
           error.code === 'ECONNABORTED' ||
           isNetworkError(error);
}

module.exports = {
    safe_llm_invoke,
    calculateMessagesSize,
    isTimeoutError,
    isNetworkError,
    isConnectionError
};
