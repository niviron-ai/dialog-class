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
        const rawConfig = arguments.length > 0 ? (arguments[0] ?? {}) : {};
        const temperatureWasProvided = Object.prototype.hasOwnProperty.call(rawConfig, 'temperature');
        const modelNameWasProvided = Object.prototype.hasOwnProperty.call(rawConfig, 'modelName');
        const providerWasProvided = Object.prototype.hasOwnProperty.call(rawConfig, 'provider');
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

        const modelResolutionTrace = {
            directParam: modelName ?? null,
            providerEnv: providerSpecificModel ?? null,
            globalEnv: process.env.DEFAULT_LLM_MODEL ?? null,
            providerDefault: this.DEFAULT_MODELS[normalizedProvider] ?? null,
            openaiDefault: this.DEFAULT_MODELS.openai
        };

        console.info('[LLM_PROVIDER_FACTORY] Resolved base configuration', JSON.stringify({
            requestedProvider,
            normalizedProvider,
            providerWasProvided,
            modelNameWasProvided,
            temperatureWasProvided,
            resolvedModel: actualModelName,
            temperature,
            modelResolutionTrace
        }));

        switch (normalizedProvider) {
            case 'openai': {
                const openAIModelsWithoutTemperature = new Set([
                    'gpt-5',
                    'gpt-5-mini'
                ]);

                const openAIOptions = {
                    apiKey: process.env.OPENAI_API_KEY,
                    modelName: actualModelName,
                    configuration: {
                        basePath: process.env.PROXY_URL,
                        baseURL: process.env.PROXY_URL
                    }
                };

                if (openAIModelsWithoutTemperature.has(actualModelName)) {
                    if (typeof temperature !== 'undefined') {
                        console.info('[LLM_PROVIDER_FACTORY][OPENAI] Ignoring temperature for model without temperature support', JSON.stringify({
                            model: actualModelName,
                            requestedTemperature: temperature
                        }));
                    }
                } else {
                    openAIOptions.temperature = temperature;
                }

                const loggableOpenAIOptions = {
                    model: openAIOptions.modelName ?? openAIOptions.model ?? null,
                    temperature: Object.prototype.hasOwnProperty.call(openAIOptions, 'temperature') ? openAIOptions.temperature : 'not-set',
                    basePath: openAIOptions.configuration?.basePath ?? null,
                    baseURL: openAIOptions.configuration?.baseURL ?? null,
                    hasApiKey: Boolean(openAIOptions.apiKey)
                };
                console.info('[LLM_PROVIDER_FACTORY][OPENAI] Final options snapshot', JSON.stringify({
                    resolvedModel: loggableOpenAIOptions.model,
                    temperature: loggableOpenAIOptions.temperature,
                    basePath: loggableOpenAIOptions.basePath,
                    baseURL: loggableOpenAIOptions.baseURL,
                    hasApiKey: loggableOpenAIOptions.hasApiKey
                }));

                return new ChatOpenAI(openAIOptions);
            }

            case 'anthropic':
                {
                const anthropicBaseUrl = process.env.ANTHROPIC_BASE_URL || process.env.PROXY_URL;
                const anthropicOptions = {
                    apiKey: process.env.ANTHROPIC_API_KEY,
                    modelName: actualModelName,
                    temperature,
                    ...(anthropicBaseUrl ? { baseUrl: anthropicBaseUrl } : {})
                };

                console.info('[LLM_PROVIDER_FACTORY][ANTHROPIC] Final options snapshot', JSON.stringify({
                    resolvedModel: anthropicOptions.modelName,
                    temperature: anthropicOptions.temperature,
                    baseUrl: anthropicBaseUrl ?? null,
                    hasApiKey: Boolean(anthropicOptions.apiKey)
                }));

                return new ChatAnthropic(anthropicOptions);
                }

            case 'yandex':
                // YandexGPT использует ChatOpenAI с OpenAI-совместимым API
                const ycFolderId = process.env.YC_FOLDER_ID;
                const ycModel = actualModelName;
                
                if (!ycFolderId) {
                    throw new Error('YC_FOLDER_ID environment variable is required for YandexGPT provider');
                }
                if (!process.env.YC_API_KEY) {
                    throw new Error('YC_API_KEY environment variable is required for YandexGPT provider');
                }

                // Формируем строку модели
                const yandexModelString = ycModel.startsWith('gpt://')
                    ? ycModel
                    : `gpt://${ycFolderId}/${ycModel}/latest`;

                const yandexBaseUrl = process.env.YC_BASE_URL || process.env.YC_API_BASE_URL || 'https://llm.api.cloud.yandex.net/v1';

                console.info('[LLM_PROVIDER_FACTORY][YANDEX] Creating ChatOpenAI instance', JSON.stringify({
                    requestedModel: modelName,
                    resolvedModel: ycModel,
                    modelUri: yandexModelString,
                    folderId: ycFolderId,
                    baseURL: yandexBaseUrl,
                    temperature,
                    hasApiKey: Boolean(process.env.YC_API_KEY)
                }));

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

        const summaryResolutionTrace = {
            globalSummaryEnv: process.env.DEFAULT_SUMMARY_MODEL ?? null,
            providerSummaryEnv: providerSpecificSummary ?? null,
            providerDefault: this.DEFAULT_SUMMARY_MODELS[normalizedProvider] ?? null,
            openaiDefault: this.DEFAULT_SUMMARY_MODELS.openai
        };

        let resolvedSummaryModel;

        if (process.env.DEFAULT_SUMMARY_MODEL) {
            resolvedSummaryModel = process.env.DEFAULT_SUMMARY_MODEL;
        } else if (providerSpecificSummary) {
            resolvedSummaryModel = providerSpecificSummary;
        } else if (this.DEFAULT_SUMMARY_MODELS[normalizedProvider]) {
            resolvedSummaryModel = this.DEFAULT_SUMMARY_MODELS[normalizedProvider];
        } else {
            resolvedSummaryModel = this.DEFAULT_SUMMARY_MODELS.openai;
        }

        console.info('[LLM_PROVIDER_FACTORY] Resolved summary model', JSON.stringify({
            requestedProvider: provider ?? null,
            normalizedProvider,
            resolvedSummaryModel,
            summaryResolutionTrace
        }));

        return resolvedSummaryModel;
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

