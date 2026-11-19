"use strict";

const locales = {
  en_US: "English", ru_RU: "Русский", es_ES: "Español", fr_FR: "Français",
  it_IT: "Italiano", ja_JP: "日本語", ko_KR: "한국어", pt_BR: "Português",
  tr_TR: "Türkçe", zh_CN: "简体中文", zh_SG: "简体中文（新加坡）",
  zh_HK: "繁體中文（香港）", zh_TW: "繁體中文（台灣）",
};

// Existing interface terms reuse the same messages as the Python WebUI.
const originalKeys = {
  "Conversion": "模型推理", "Training": "模型训练", "Help": "常见问题解答",
  "Models": "ckpt处理", "Voice model": "推理音色", "Speaker ID": "请选择说话人id",
  "Exclusive WASAPI": "独占 WASAPI 设备", "Device rate": "使用设备采样率",
  "Multi-speaker dataset helper": "多说话人训练集辅助",
  "Train index": "训练特征索引", "Save speaker manifest": "提交训练集清单",
  "Train prepared model": "训练模型", "Prepare, train and index": "一键训练",
  "Pretrained generator": "加载预训练底模G路径", "Pretrained discriminator": "加载预训练底模D路径",
  "Retrieval index": "索引",
};

const russian = {
  "Language": "Язык",
  "Conversion": "Конвертация", "Realtime": "Реалтайм", "Training": "Обучение",
  "Separation": "Разделение аудио", "Models": "Модели", "Help": "Справка",
  "Voice conversion": "Конвертация голоса", "Source audio": "Исходное аудио",
  "Convert audio": "Конвертировать", "Converted audio": "Результат",
  "Unload voice model": "Выгрузить модель",
  "Download WAV": "Скачать WAV", "Batch conversion": "Пакетная конвертация",
  "Source folder": "Исходная папка", "Output folder": "Папка результатов",
  "Output format": "Формат результата", "Include subfolders": "Включить подпапки",
  "Replace existing files": "Заменять существующие файлы", "Convert folder": "Конвертировать папку",
  "Realtime conversion": "Конвертация в реальном времени", "Model file path": "Путь к файлу модели",
  "Audio host": "Аудиосистема", "Input device": "Устройство ввода", "Output device": "Устройство вывода",
  "Sample rate": "Частота дискретизации", "Device rate": "Частота устройства", "Model rate": "Частота модели",
  "Block (ms)": "Блок (мс)", "Crossfade (ms)": "Кроссфейд (мс)", "Context (ms)": "Контекст (мс)",
  "Silence threshold (dB)": "Порог тишины (дБ)", "Formant shift": "Сдвиг формант",
  "Microphone monitor": "Прослушивание микрофона", "Exclusive WASAPI": "Эксклюзивный WASAPI",
  "Input noise reduction": "Шумоподавление на входе", "Output noise reduction": "Шумоподавление на выходе",
  "Start audio": "Запустить аудио", "Stop audio": "Остановить аудио",
  "Prepare dataset": "Подготовка датасета", "Dataset mode": "Тип датасета",
  "Single speaker": "Один диктор", "Multi-speaker folders": "Папки дикторов",
  "Saved speaker manifest": "Сохранённый манифест", "Dataset folder": "Папка датасета",
  "Experiment folder": "Папка эксперимента", "Workers": "Рабочие потоки",
  "Segment length (s)": "Длительность сегмента (с)", "Prepare audio": "Подготовить аудио",
  "Extract features": "Извлечь признаки", "Feature extraction": "Извлечение признаков",
  "Extraction": "Извлечение", "Compute device": "Устройство вычислений", "Model version": "Версия модели",
  "CUDA devices": "Устройства CUDA", "Pitch extraction": "Извлечение F0",
  "No pitch extraction": "Пропустить извлечение F0", "Recompute cached features": "Пересчитать сохранённые признаки",
  "Train retrieval index": "Построение индекса поиска",
  "Pitch estimator": "Алгоритм оценки F0", "Pitch guidance": "Управление высотой тона (F0)",
  "Train voice model": "Обучение голосовой модели", "Output model name": "Имя модели",
  "Epochs": "Эпохи", "Batch size": "Размер батча", "Save every (epochs)": "Сохранять каждые (эпох)",
  "Speaker ID": "ID диктора", "Multi-speaker training": "Несколько дикторов",
  "Checkpoints and pretrained weights": "Контрольные точки и предобученные веса",
  "Pretrained generator": "Предобученный генератор", "Pretrained discriminator": "Предобученный дискриминатор",
  "Export intermediate models": "Экспортировать промежуточные модели",
  "Keep one rolling checkpoint pair": "Хранить одну пару контрольных точек",
  "Initialize without default pretrained weights": "Начать без предобученных весов",
  "Training precision": "Точность обучения", "Cache batches on GPU": "Кэшировать батчи в VRAM",
  "Write TensorBoard logs": "Записывать события TensorBoard", "Train prepared model": "Начать обучение",
  "Run every training stage": "Запуск всех этапов", "Prepare, train and index": "Подготовить, обучить и построить индекс",
  "Index training": "Построение индекса", "Train index": "Построить индекс",
  "Multi-speaker dataset helper": "Манифест дикторов", "Save speaker manifest": "Сохранить манифест",
  "Add speaker": "Добавить диктора", "Remove speaker": "Удалить диктора", "Repeat": "Повторы",
  "Speaker name": "Имя диктора", "Speaker folder": "Папка диктора",
  "Separation model": "Модель разделения", "Separation preset": "Режим разделения",
  "Remove reverb": "Удалить реверберацию", "Remove reverb (aggressive)": "Усиленное удаление реверберации",
  "Separate vocals and accompaniment": "Вокал и аккомпанемент",
  "Separate vocals and accompaniment (aggressive)": "Вокал и аккомпанемент, усиленный режим",
  "Main and backing vocals": "Основной и бэк-вокал", "Separate audio": "Разделить аудио",
  "Model details": "Данные модели", "Refresh models": "Обновить список моделей",
  "Edit description": "Изменить описание", "Description": "Описание", "Save description": "Сохранить описание",
  "Merge models": "Объединить модели", "First model": "Первая модель", "Second model": "Вторая модель",
  "Weight of the first model": "Доля первой модели", "Output filename": "Имя файла результата",
  "Replace existing file": "Заменить существующий файл", "Export inference weights": "Экспорт модели для конвертации",
  "Training checkpoint": "Контрольная точка обучения", "Export model": "Экспортировать",
  "Background task": "Фоновая задача", "Cancel task": "Отменить",
  "Voice model": "Голосовая модель", "Speaker": "Диктор", "Automatic": "Автоматически",
  "Pitch shift (semitones)": "Сдвиг тона (полутоны)", "Conversion settings": "Настройки конвертации",
  "Retrieval index": "Файл индекса", "Index mix": "Доля поиска по индексу",
  "Volume envelope mix": "Доля огибающей громкости", "Consonant protection": "Защита согласных",
  "Output rate (Hz)": "Частота результата (Гц)", "System default": "Системное устройство",
  "Select a model": "Выберите модель", "Select a voice model.": "Выберите голосовую модель.",
  "No models in assets/weights": "В assets/weights нет моделей", "Converting audio…": "Конвертация аудио…",
  "Loading model and audio devices…": "Загрузка модели и аудиоустройств…",
  "Running": "Выполняется", "Complete": "Готово", "Failed": "Ошибка", "Cancelled": "Отменено",
  "Cancellation requested…": "Отмена задачи…",
  "Where are the voice models?": "Где лежат голосовые модели?",
  "Put compact .pth voice models in assets/weights and open Models to refresh the list. Training checkpoints in logs can be exported from Models.": "Поместите компактные модели .pth в assets/weights и обновите список в разделе «Модели». Там же можно экспортировать модель из контрольной точки обучения в logs.",
  "What does the retrieval index do?": "Как использовать индекс?",
  "Use the added .index file for the model and speaker. Set Index mix to zero for conversion without retrieval.": "Выберите файл added .index для нужной модели и диктора. При доле поиска 0 конвертация идёт без индекса.",
  "How do I train a voice?": "Как обучить голосовую модель?",
  "Prepare the recordings, extract features and pitch, train, then build the index. The one-step button runs these stages with the settings in the training form.": "Подготовьте записи, извлеките признаки и высоту тона, обучите модель и постройте индекс. Кнопка запуска всех этапов использует настройки из формы обучения.",
  "Why does realtime audio break up?": "Почему прерывается звук в реальном времени?",
  "The status shows processing time per block and audio dropouts. Increase the block duration if processing regularly takes longer than a block.": "В строке состояния видны время обработки блока и пропуски аудио. Увеличьте длительность блока, если обработка регулярно занимает больше времени.",
};

let messages = {};
const sourceText = new WeakMap();
const t = english => Object.hasOwn(messages, english) ? messages[english] : english;

function translate(root = document.body) {
  const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let node = walk.nextNode(); node; node = walk.nextNode()) {
    if (node.parentElement?.closest("script, style, pre, #language")) continue;
    if (!sourceText.has(node)) sourceText.set(node, node.nodeValue);
    const original = sourceText.get(node);
    const key = original.trim();
    if (key) node.nodeValue = original.replace(key, t(key));
  }
}

async function switchLanguage(language) {
  const translations = {};
  if (language !== "en_US" && language !== "ru_RU") {
    const response = await fetch("/api/locale/" + encodeURIComponent(language));
    if (!response.ok) throw new Error("Cannot load interface language");
    const original = await response.json();
    for (const [english, key] of Object.entries(originalKeys)) {
      if (original[key]) translations[english] = original[key];
    }
  }
  messages = language === "ru_RU" ? russian : translations;
  document.documentElement.lang = language.replace("_", "-");
  translate();
}

function interfaceLanguage(report) {
  const select = document.getElementById("language");
  const stored = localStorage.getItem("rvc-language");
  const preferred = navigator.languages.flatMap(name => {
    const code = name.replace("-", "_");
    return Object.hasOwn(locales, code) ? [code]
      : Object.keys(locales).filter(locale => locale.startsWith(name.slice(0, 2) + "_"));
  })[0];
  select.replaceChildren(...Object.entries(locales).map(([code, name]) => new Option(name, code)));
  select.value = stored && Object.hasOwn(locales, stored) ? stored : preferred ?? "en_US";
  select.addEventListener("change", () => {
    localStorage.setItem("rvc-language", select.value);
    switchLanguage(select.value).catch(report);
  });
  return switchLanguage(select.value);
}
