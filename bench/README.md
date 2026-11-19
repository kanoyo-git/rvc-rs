# Сравнение производительности RVC

`compare.py` запускает оригинальный Python CLI и Rust CLI на общих весах и WAV. В варианте `single` каждый процесс конвертирует один файл. В варианте `batch` один процесс конвертирует всю папку и повторно использует загруженные модели.

Подготовьте отдельную рабочую копию исходного RVC на коммите `81eed5e8f68b6bed1789f682fe78cdd324495afc` с окружением Python и моделями HuBERT/RMVPE. Соберите Rust CLI в профиле `release` с совместимым LibTorch. Измеритель требует Linux и Python 3.11+. Для примера ниже используются два логических CPU с номерами 4 и 5, их можно заменить на свободные CPU своего компьютера. Генераторы эталонных данных для тестов PyMSS находятся в `bench/fixtures/` и читают исходники из отдельной рабочей копии. [Исходный репозиторий и версия](../THIRD_PARTY.md#retrieval-based-voice-conversion-webui)

```bash
taskset -c 4,5 python3 bench/compare.py \
  --baseline /path/to/original \
  --python /path/to/original/.venv/bin/python \
  --rust-binary target/release/rvc \
  --rust-lib-dir /path/to/libtorch/lib \
  --model /path/to/voice.pth \
  --inputs /path/to/pcm-wavs \
  --threads 2 --runs 5 --warmups 1 \
  --json /tmp/rvc-cpu-benchmark.json
```

- Измеритель чередует порядок Python и Rust, привязывает оба процесса к одним CPU и задаёт одинаковые лимиты потоков. Прогрев выполняется отдельно для каждого сценария. Каждый замер запускает новый процесс, с обычным кешем файлов ОС.
- Время включает запуск CLI, чтение моделей, декодирование, конвертацию и запись WAV. CPU user/system и RSS берутся из Linux `wait4`. Пиковый RSS соответствует максимальному RSS процесса с учётом дочерних процессов, которых он дождался. В JSON коэффициент загрузки CPU считается из времени CPU и стены, где 100% соответствует одному логическому ядру.
- Измеритель сверяет имена, число сэмплов, частоты, разрядность и каналы выходных WAV. В JSON сохраняются отдельные замеры, медианы, версии сред исполнения и SHA-256 весов и входов.
- Для CUDA используйте CUDA-сборки обоих окружений и добавьте `--device cuda --gpu 0 --gpu-stats`. VRAM привязана к PID, а загрузка GPU относится ко всему устройству. Обе величины снимаются опросом `nvidia-smi`, среднее и максимум зависят от частоты выборки. Автоматически выбранная точность Python отражается в JSON.
- Retrieval выключен по умолчанию. Для него передайте одинаковый индекс через `--index` и задайте смесь в `--index-rate`. При необходимости укажите общий `--speaker-id`.

## WebUI

`web_compare.py` запускает оба WebUI на loopback в отдельных временных каталогах и отправляет одинаковые запросы Gradio. В конфигурации Python должна быть Gradio 3.14. Каждый прогон запускает сервер, один раз выбирает модель и затем пять раз загружает и конвертирует один WAV. Измеритель сверяет схему вызовов и параметры выходных WAV.

```bash
taskset -c 4,5 python3 bench/web_compare.py \
  --baseline /path/to/original --python /path/to/original/.venv/bin/python \
  --rust-binary target/release/rvc --rust-lib-dir /path/to/libtorch/lib \
  --model /path/to/voice.pth --input /path/to/source.wav \
  --threads 2 --runs 3 --calls 5 --json /tmp/rvc-web-benchmark.json
```

## Реалтайм

Соберите Rust-бенчмарк с теми же опциями, что и проверяемое приложение. Для CUDA-графов добавьте `--features cuda-graph` и используйте заголовки CUDA Toolkit.

```bash
cargo build --release --locked -p rvc --example realtime_benchmark
ffmpeg -i source.wav -ac 1 -ar 16000 -c:a pcm_f32le -f f32le /tmp/rvc-stream.f32
taskset -c 4,5 python3 bench/realtime_compare.py \
  --baseline /path/to/original --python /path/to/original/.venv/bin/python \
  --rust-binary target/release/examples/realtime_benchmark \
  --rust-lib-dir /path/to/libtorch/lib --model voice.pth --input /tmp/rvc-stream.f32 \
  --device cuda --gpu 0 --blocks 60 --warmups 5 --threads 2 \
  --json /tmp/rvc-realtime-benchmark.json
```

Оба движка получают блоки по 250 мс и 2,5 с контекста. Измеряется обработка блока моделью. Старт процесса и прогрев учитываются в измерении ресурсов. Задержка устройств и аудиобуферов проверяется отдельным прогоном.

## Подготовка и обучение

`pipeline_compare.py` запускает подготовку аудио, извлечение HuBERT, построение FAISS и тренировку одной эпохи. Для последнего этапа нужен эксперимент v2 32 кГц без F0 с `config.json`, `filelist.txt` в формате `wav|feature|speaker` и соответствующими WAV и NPY. Если манифест ссылается на тишину из оригинального проекта, её файлы должны быть в `logs/mute/`. Исходные данные копируются во временные каталоги.

```bash
taskset -c 4,5 python3 bench/pipeline_compare.py \
  --baseline /path/to/original --python /path/to/original/.venv/bin/python \
  --rust-binary target/release/rvc --rust-lib-dir /path/to/libtorch/lib \
  --inputs /path/to/pcm-wavs --training-fixture /path/to/original/logs/experiment \
  --threads 2 --runs 3 --json /tmp/rvc-pipeline-benchmark.json
```

Для CUDA используйте совместимые GPU-сборки PyTorch и LibTorch и добавьте `--device cuda --gpu 0 --case extract --case train`. Номер GPU соответствует порядку PCI. VRAM опрашивается по PID. Замер обучения включает запись обоих чекпойнтов и экспорт компактной модели.

Чтобы оба процесса использовали одни и те же бинарные библиотеки Torch, передайте `--rust-lib-dir /path/to/venv/lib/python3.12/site-packages/torch/lib`. Версия Torch, вариант CUDA и C++ ABI пакета должны совпадать с параметрами сборки Rust. Замер Rust по-прежнему запускает непосредственно `rvc`.

## Разделение аудио

`pymss_compare.py` измеряет один трек, используя одинаковые каталожные веса и настройки. `ffprobe` сверяет имена файлов, частоты, каналы, кодек и длительность WAV после каждого прогона.

```bash
taskset -c 4,5 python3 bench/pymss_compare.py \
  --baseline /path/to/original --python /path/to/original/.venv/bin/python \
  --rust-binary target/release/rvc --rust-lib-dir /path/to/libtorch/lib \
  --model-dir /path/to/model-dir --model HTDemucs4.th --input stereo.wav \
  --threads 2 --runs 3 --json /tmp/rvc-pymss-benchmark.json
```

Для CUDA укажите CUDA-сборки Python и Rust и добавьте `--device cuda --gpu 0 --gpu-stats`. `--param KEY=VALUE` передаёт обеим версиям одну настройку инференса. Флаг можно повторить для нескольких параметров. VRAM опрашивается по PID каждые 200 мс.
