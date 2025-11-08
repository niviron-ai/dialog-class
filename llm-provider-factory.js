const { ChatOpenAI } = require('@langchain/openai');
const { ChatAnthropic } = require("@langchain/anthropic");
const { convertToOpenAITool } = require("@langchain/core/utils/function_calling");

/**
 * Фабрика для создания LLM провайдеров
 * Поддерживает OpenAI, Anthropic и YandexGPT (через OpenAI-совместимый API)
 */
class LLMProviderFactory {
    /**
     * Дефолтные модели для каждого провайдера
     */
    static DEFAULT_MODELS = {
        openai: 'gpt-5',
        anthropic: 'claude-sonnet-4-5',
        yandex: 'gpt-oss-120b'
    };

    /**
     * Дефолтные модели для суммаризации (более дешевые)
     */
    static DEFAULT_SUMMARY_MODELS = {
        openai: 'gpt-5-mini',
        anthropic: 'claude-haiku-4-5',
        yandex: 'gpt-oss-20b'
    };

    /**
     * Создает экземпляр LLM для указанного провайдера
     * @param {Object} config - Конфигурация LLM
     * @param {string} config.modelName - Название модели
     * @param {number} config.temperature - Температура генерации
     * @param {string} config.provider - Провайдер ('openai', 'anthropic', 'yandex')
     * @returns {ChatOpenAI|ChatAnthropic} Экземпляр LLM
     */
    static create({modelName, temperature = 0, provider} = {}) {
        // Определяем провайдер из параметра, переменной окружения или дефолтного значения
        const requestedProvider = provider ?? process.env.DEFAULT_LLM_PROVIDER ?? 'openai';
        const normalizedProvider = String(requestedProvider).toLowerCase();
        
        // Определяем модель из переменных окружения, если не передана
        const providerEnvKey = `DEFAULT_LLM_MODEL_${normalizedProvider.toUpperCase()}`;
        const providerSpecificModel = process.env[providerEnvKey];
        const actualModelName = modelName ||
            providerSpecificModel ||
            process.env.DEFAULT_LLM_MODEL ||
            this.DEFAULT_MODELS[normalizedProvider] ||
            this.DEFAULT_MODELS.openai;

        switch (normalizedProvider) {
            case 'openai':
                return new ChatOpenAI({
                    apiKey: process.env.OPENAI_API_KEY,
                    modelName: actualModelName,
                    temperature,
                    configuration: {
                        basePath: process.env.PROXY_URL,
                        baseURL: process.env.PROXY_URL
                    }
                });

            case 'anthropic':
                {
                const anthropicBaseUrl = process.env.ANTHROPIC_BASE_URL || process.env.PROXY_URL;
                return new ChatAnthropic({
                    apiKey: process.env.ANTHROPIC_API_KEY,
                    modelName: actualModelName,
                    temperature,
                    ...(anthropicBaseUrl ? { baseUrl: anthropicBaseUrl } : {})
                });
                }

            case 'yandex':
                // YandexGPT использует ChatOpenAI с OpenAI-совместимым API
                const ycFolderId = process.env.YC_FOLDER_ID;
                // Используем YC_MODEL из переменных окружения, если задан, иначе переданную модель или дефолтную
                const ycModel = process.env.YC_MODEL || actualModelName;
                
                if (!ycFolderId) {
                    throw new Error('YC_FOLDER_ID environment variable is required for YandexGPT provider');
                }
                if (!process.env.YC_API_KEY) {
                    throw new Error('YC_API_KEY environment variable is required for YandexGPT provider');
                }

                // Формируем строку модели в формате gpt://{folder}/{model}/latest
                const yandexModelString = `gpt://${ycFolderId}/${ycModel}/latest`;

                const yandexBaseUrl = process.env.YC_BASE_URL || process.env.YC_API_BASE_URL || process.env.PROXY_URL || 'https://llm.api.cloud.yandex.net/v1';

                return new ChatOpenAI({
                    apiKey: process.env.YC_API_KEY,
                    model: yandexModelString,
                    temperature,
                    configuration: {
                        baseURL: yandexBaseUrl
                    }
                });

            default:
                throw new Error(`Unsupported LLM provider: ${requestedProvider}. Supported providers: openai, anthropic, yandex`);
        }
    }

    /**
     * Подготавливает инструменты для привязки к LLM
     * Для OpenAI и YandexGPT применяет convertToOpenAITool
     * Для Anthropic возвращает инструменты как есть
     * @param {Array} tools - Массив инструментов
     * @param {string} provider - Провайдер ('openai', 'anthropic', 'yandex')
     * @returns {Array} Подготовленные инструменты
     */
    static prepareToolsForBinding(tools, provider) {
        if (!tools || tools.length === 0) {
            return tools;
        }

        // OpenAI и YandexGPT используют ChatOpenAI, поэтому нужна конвертация
        const normalizedProvider = String(provider || '').toLowerCase();
        if (normalizedProvider === 'openai' || normalizedProvider === 'yandex') {
            return tools.map(tool => convertToOpenAITool(tool));
        }

        // Anthropic и другие провайдеры - инструменты передаем как есть
        return tools;
    }

    /**
     * Получает модель для суммаризации для указанного провайдера
     * @param {string} provider - Провайдер
     * @returns {string} Название модели для суммаризации
     */
    static getSummaryModel(provider) {
        const normalizedProvider = String(provider || '').toLowerCase();
        const providerEnvKey = `DEFAULT_SUMMARY_MODEL_${normalizedProvider.toUpperCase()}`;
        const providerSpecificSummary = process.env[providerEnvKey];
        return process.env.DEFAULT_SUMMARY_MODEL ||
            providerSpecificSummary ||
            this.DEFAULT_SUMMARY_MODELS[normalizedProvider] ||
            this.DEFAULT_SUMMARY_MODELS.openai;
    }

    /**
     * Получает список поддерживаемых провайдеров
     * @returns {Array<string>} Массив названий провайдеров
     */
    static getSupportedProviders() {
        return Object.keys(this.DEFAULT_MODELS);
    }
}

module.exports = { LLMProviderFactory };

