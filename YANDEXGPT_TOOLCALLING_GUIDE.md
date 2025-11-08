## Руководство: интеграция LangChain ↔ YandexGPT (OpenAI‑совместимый API)

Документ фиксирует фактические настройки и последовательность действий, которые подтвердили работоспособность моделей Яндекс Облака в связке с LangChain. Цель — передать параметры и наблюдения; архитектурные решения остаются на усмотрение команды проекта.

---

### 1. Зависимости
```
npm install @langchain/openai @langchain/core langchain zod dotenv
```

---

### 2. Переменные окружения
Минимальный набор:

```
YC_API_KEY=...    # API-ключ сервисного аккаунта (роль ai.languageModels.user)
YC_FOLDER_ID=...  # идентификатор каталога
YC_MODEL=...      # короткое имя модели, например gpt-oss-120b
```

Из этих значений формируется строка модели:
```
gpt://{YC_FOLDER_ID}/{YC_MODEL}/latest
```

Базовый URL по умолчанию — `https://llm.api.cloud.yandex.net/v1` (OpenAI‑совместимый endpoint).

---

### 3. Создание экземпляра `ChatOpenAI`
Пример параметров, которые отработали в песочнице:

```javascript
import { ChatOpenAI } from '@langchain/openai';

const client = new ChatOpenAI({
  apiKey: process.env.YC_API_KEY,
  model: `gpt://${process.env.YC_FOLDER_ID}/${process.env.YC_MODEL}/latest`,
  temperature: 0.7,
  configuration: {
    baseURL: 'https://llm.api.cloud.yandex.net/v1', // можно не указывать, если подходит значение по умолчанию
  },
});
```

### 4. Рабочие сценарии использования
#### 4.1 Простой чат
```javascript
const answer = await client.invoke('Привет! Ты работаешь?');
console.log(answer.content);
```

#### 4.2 Tool Calling (LangChain tools + zod)
```javascript
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

const get_weather = tool(
  async ({ city }) => lookup_weather(city),
  {
    name: 'get_weather',
    description: 'Получить текущую погоду',
    schema: z.object({
      city: z.string().describe('Название города'),
    }),
  },
);

const clientWithTools = client.bindTools([get_weather]);
const response = await clientWithTools.invoke('Какая погода в Москве?');

if (response.tool_calls?.length) {
  const result = await get_weather.invoke(response.tool_calls[0].args);
  // дальше — на усмотрение проекта (например, добавить ToolMessage и продолжить диалог)
}
```



### 5. Наблюдения по tool calling
- Инструменты описывались через `tool()` + `zod`. LangChain автоматически преобразует схему в JSON Schema.
- Метод `bindTools()` поддерживает опции `strict`, `tool_choice`, `parallel_tool_calls`.
- При tool call поле `content` в `AIMessage` может быть `undefined`; данные находятся в `response.tool_calls`.
- Для агентских сценариев после выполнения инструмента стоит добавлять `ToolMessage` с результатом обратно в историю.

---


### 6. Полезные ссылки
- OpenAI‑совместимый API: https://yandex.cloud/ru/docs/ai-studio/concepts/openai-compatibility
- Tool calling в LangChain: https://js.langchain.com/docs/modules/model_io/chat/function_calling
- Каталог моделей Foundation Models: https://yandex.cloud/ru/docs/ai-studio/concepts/generation/models

---

### 7. Итог
- Рабочая конфигурация сводится к `ChatOpenAI` с параметрами `apiKey` и `model = gpt://{folder}/{model}/latest`.
- Базовый endpoint — `https://llm.api.cloud.yandex.net/v1`.
- LangChain возможности (tool calling, управление историей, structured output) функционируют без дополнительных адаптаций.
