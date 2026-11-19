# RVC-RS

Рабочая область Cargo содержит приложение `rvc` и плагин `rvc-plugin`. Приложение обрабатывает совместимые модели RVC v1/v2, индексы FAISS и каталоги PyMSS. Для запуска нужен LibTorch, для аудио нужны FFmpeg и FFprobe, для загрузки каталожных моделей PyMSS нужен `curl`.

Ссылка на исходный RVC, эталонный коммит и лицензии приведены в [THIRD_PARTY.md](THIRD_PARTY.md).

## Структура

| Каталог | Содержимое |
| --- | --- |
| `crates/rvc/` | CLI, WebUI, обучение, конвертация, реалтайм и PyMSS |
| `crates/rvc-plugin/` | Плагин VST2/VST3 и примеры для его проверки в хосте |
| `assets/` и `configs/` | Веса, индексы и конфигурации моделей |
| `bench/` | Отчёт, сравнительные замеры и генераторы тестовых данных |
| `licenses/` | Лицензии и атрибуция использованных ресурсов и исходных компонентов |

## Сборка

Установите Rust, C++-компилятор, CMake и BLAS. Для Linux также нужны заголовки ALSA, библиотеки X11/OpenGL для редактора плагина и `pkg-config`. Версия LibTorch должна быть 2.7.0, как у `tch` из [Cargo.lock](Cargo.lock). Выберите CPU-сборку LibTorch или сборку под свой CUDA-драйвер и укажите её каталог.

```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH="$LIBTORCH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo build --release --locked --workspace
cargo test --locked --workspace
```

В Windows добавьте `%LIBTORCH%\lib` и каталог FFmpeg в `PATH`. `go-webui.bat` и `go-realtime_gui.bat` запускают `rvc.exe` из корня проекта или из `target\release`. Пакетирование VST описано в [README плагина](crates/rvc-plugin/README.md).

Чтение индексов для конвертации встроено в приложение. Для построения новых индексов на Linux Cargo собирает FAISS статически. На Windows и macOS используется динамическая библиотека FAISS с C API. Соберите FAISS 1.12.0 с `-DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF` и включите `-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON` для Windows. Укажите каталог линковки через `FAISS_LIB_DIR`. Добавьте каталог DLL в `PATH` на Windows. Для локальной библиотеки на macOS укажите её каталог в `DYLD_LIBRARY_PATH`. Сборка с `--no-default-features --features inference,realtime` сохраняет чтение индексов. [FAISS 1.12.0](https://github.com/facebookresearch/faiss/tree/v1.12.0).

Для CUDA-реалтайма с графами используйте сборку LibTorch с CUDA и заголовки совместимого CUDA Toolkit. Укажите корень Toolkit через `CUDA_HOME` или `CUDA_PATH`. Если заголовки находятся в нескольких каталогах, перечислите их в `CUDA_INCLUDE_DIRS` через системный разделитель путей.

```bash
cargo build --release --locked --workspace --features rvc/cuda-graph
```

Для запуска Linux CUDA-сборки с меньшим пиковым RSS можно использовать нативные библиотеки из пакетов PyTorch 2.7.0+cu128. При распаковке сохраните относительные пути `torch/lib`, `nvidia/*/lib` и `cusparselt/lib`. Укажите `torch/lib` в `LD_LIBRARY_PATH`. В таком наборе достаточно файлов `.so`; запуск проверен без `libtorch_python.so` и с отдельным деревом нативных библиотек. [Замеры памяти](bench/RESULTS.md).

HuBERT, оценка F0 и генератор переиспользуют память между последовательными захватами. При смене геометрии аудиоблока графы обновляются. При изменении формант во время работы генератор сразу обрабатывает новую форму в обычном режиме CUDA. При возврате к исходной форме снова используется CUDA-граф. `RVC_CUDA_GRAPH=0` включает прямое выполнение CUDA. Для ASIO в Windows используйте `--features rvc/asio` с установленным драйвером, LLVM и SDK в `CPAL_ASIO_DIR`.

## Модели

Поместите файлы оригинального набора весов в следующие каталоги.

| Назначение | Путь относительно корня проекта |
| --- | --- |
| ContentVec для конвертации и извлечения признаков | `assets/hubert_base/pytorch_model.bin` |
| Настройки нормализации при извлечении признаков | `assets/hubert_base/preprocessor_config.json` |
| RMVPE | `assets/rmvpe/rmvpe.pt` |
| FCPE | `assets/fcpe/fcpe_c_v001.pt` из пакета `torchfcpe 0.0.4` |
| Компактные голосовые модели | `assets/weights/*.pth` |
| Готовые индексы | `assets/indices/` или каталог эксперимента в `logs/` |
| Предобученные генераторы и дискриминаторы | `assets/pretrained/` и `assets/pretrained_v2/` |
| Образцы тишины для создания `filelist.txt` | `logs/mute/` после распаковки `mute.zip` в `logs/` |

RMVPE и предобученные веса доступны на зеркале [Politrees/RVC_resources](https://huggingface.co/Politrees/RVC_resources).

## Запуск

Запускайте команды из корня проекта. Для отдельного каталога моделей и конфигурации используйте глобальный аргумент `--root`.

```bash
target/release/rvc serve --open
target/release/rvc serve --port 7898 --panel realtime --open
target/release/rvc devices
target/release/rvc infer --model voice.pth --input source.wav --output converted.wav --index-rate 0
```

В CLI доступны `scan-speakers`, `preprocess`, `extract`, `train`, `train-index`, `export-model`, `merge-models` и `set-model-info`. Запуск с `--help` показывает параметры команд. `train` использует существующий `filelist.txt`. Флаг `--rebuild-filelist` пересоздаёт список из подготовленных данных. Обучение продолжает последнюю пару `G_*.pth` и `D_*.pth` из эксперимента, включая расширение таблицы дикторов с сохранением прежних строк. После прерванной замены чекпойнтов следующий `train` завершает сохранение пары перед возобновлением обучения. `train` дописывает параметры и ход обучения в `train.log` и поддерживает TensorBoard, CUDA FP16, кэш датасета на GPU и выбор GPU через `--gpus`. `extract` принимает такой же список устройств.

WebUI по умолчанию слушает loopback и включает загрузку аудио, прослушивание результата, пакетную конвертацию, настройки аудиоустройств, подготовку датасетов, обучение и экспорт моделей. При первом запуске раздел realtime читает прежние настройки из `configs/config.json`. Новый профиль сохраняется в `configs/realtime-rust.json`. При смене RMVPE или FCPE веса загружаются в фоне и переиспользуются до остановки потока. На CUDA новый метод прогревается до переключения. Сборка с CUDA-графами выполняет прогрев в отдельном CUDA-потоке. При одновременной смене метода F0 и формант новые форманты применяются после готовности метода. Если выходная очередь временно заполнена, обработчик ждёт места для целого блока.

Для доступа из локальной сети задайте `rvc serve --listen 0.0.0.0 --port 7865` и откройте числовой IP-адрес компьютера с портом сервера. Прямой HTTP-доступ проверяет совпадение Host и Origin с адресом соединения.

Клиенты Gradio 3.14 могут использовать `/config`, `/api/{name}`, `/run/{name}` и WebSocket `/queue/join`. У обработчиков сохранены имена и числовые `fn_index` исходного WebUI. Для результатов старого аудио-API работают `/file={name}` и `/file/{name}`. Очередь совместимости выполняет задания по одному, чтобы ограничить пиковое потребление памяти при параллельных запросах.

Для запуска с внешней ссылкой нужен совместимый с Gradio FRP-клиент. Команда `rvc serve --colab --share-client /path/to/frpc` выводит HTTPS-адрес и пароль для пользователя `rvc`. Путь можно задать через `RVC_SHARE_CLIENT`, пароль через `RVC_SHARE_PASSWORD`. Запуск, авторизация, передача HTTP и завершение проверены в изолированной сети с клиентом Gradio 0.1 и совместимым релеем FRP 0.44.0. Завершение WebUI останавливает клиент на этапе запуска и при активном туннеле.

PyMSS поддерживает имена моделей из каталога и локальные файлы. Запуск `rvc pymss serve --webui` открывает отдельный интерфейс на `/ui/` с каталогом, загрузкой моделей, выбором дорожек, прослушиванием и скачиванием. Справка HTTP API доступна на `/docs` и `/redoc`, схема на `/openapi.json`. `pymss workflow` выполняет цепочки из YAML. Для проверки цепочки используйте `pymss workflow validate -c workflow.yaml`, добавив `--check-models` для проверки каталога или `--require-files` для проверки локальных файлов. `--debug` в `pymss infer`, `pymss serve` и `pymss workflow run` выводит выбранные пути моделей и параметры инференса.

```bash
target/release/rvc pymss list
target/release/rvc pymss infer HTDemucs4.th --input song.wav --output stems
target/release/rvc pymss serve --webui --port 8000
```

В `pymss infer` и `pymss serve` параметр `--device-id` принимает номер CUDA-устройства и допускает повторение. Для моделей с оконной обработкой `--param batch_size=N` задаёт количество окон в пакете. При нескольких GPU окна пакета распределяются между выбранными устройствами. VR и legacy Demucs используют первый выбранный GPU.

`--param stem_batch_size=N` собирает выбранные дорожки группами по N и ограничивает объём промежуточного PCM. Для общей нормализации дорожки обрабатываются одной группой.

Для RoFormer в сборке с CUDA LibTorch доступен `--param cuda_attention_backend=cudnn`. Также доступны `auto`, `default`, `flash`, `efficient` и `math`. Значение `xformers` направляет вычисление в ядро `efficient` из LibTorch. Когда выбранное ядро несовместимо с устройством или размером тензоров, LibTorch подбирает подходящее ядро. Если cuDNN не может построить план для слоя, режим `auto` пробует `efficient`, а режим `cudnn` передаёт выбор ядра LibTorch.

## MLX для PyMSS

В BS-RoFormer, Mel-Band-RoFormer, HyperACE, MDX23C, SCNet, Apollo, Bandit, Bandit v2 и гибридном HTDemucs MLX выполняет спектральные преобразования и инференс сети. В UVR VR MLX рассчитывает маску, спектр и аудио обрабатывает CPU. Укажите путь к библиотеке MLX C 0.8 в `RVC_MLX_LIBRARY`. Библиотека должна быть собрана с MLX 0.32.4. [MLX C 0.8](https://github.com/ml-explore/mlx-c/tree/v0.8.0).

При выборе MPS эти конфигурации используют MLX, если задан `RVC_MLX_LIBRARY`. Явный выбор доступен через `--param mps_model_backend=mlx_full`. Через `--param mps_model_compute_dtype=float32` можно включить FP32. По умолчанию на MPS используется FP16, на CPU используется FP32. В режимах сегментации HyperACE применяется FP32, поскольку понижение точности искажало дорожки даже на коротком контрольном сигнале. Обе точности проверены через Linux-версию MLX на CPU. Для замеров Metal нужен Apple Silicon.

`--device mlx` выбирает MPS с библиотекой из `RVC_MLX_LIBRARY`. На RoFormer параметр `mps_attention_backend` позволяет передать в MLX само внимание (`mlx`), весь слой внимания (`mlx_attention`) или трансформер (`mlx_transformer`). `--param mps_mlx_min_tokens=128` задаёт минимальную длину последовательности для переноса. `--param mps_mlx_clear_cache=true` освобождает кеш аллокатора после вычисления пакета окон RoFormer.

## RVC через ONNX Runtime

Сборка с `--features onnx` поддерживает ONNX Runtime с C API 24. Укажите путь к его библиотеке через `RVC_ONNX_RUNTIME`. Значение `RVC_ONNX_PROVIDER=cpu` включает этот движок для нейросетевых этапов RVC. Для DirectML на Windows используйте `RVC_ONNX_PROVIDER=directml:0` вместе с `--device cpu`, где 0 — номер адаптера, а CPU обрабатывает аудио и подготавливает входные тензоры. [ONNX Runtime DirectML 1.24.4](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/1.24.4).

Отдельные этапы настраиваются через `RVC_CONTENTVEC_ONNX`, `RVC_RMVPE_ONNX`, `RVC_FCPE_ONNX`, `RVC_PRIOR_ONNX`, `RVC_FLOW_ONNX` и `RVC_DECODER_ONNX`. Для каждого этапа можно задать `torch`, `cpu` или `directml:N`. По умолчанию используется LibTorch. Графы строятся из штатных весов и используют временные файлы ONNX с внешними тензорами. Если в каталоге RMVPE доступен только оригинальный `rmvpe.onnx`, загрузится этот файл. Если временный каталог расположен в RAM, для замеров памяти укажите каталог на диске через `TMPDIR` в Linux или `TMP` в Windows. [Веса RMVPE](https://huggingface.co/Politrees/RVC_resources/blob/main/predictors/rmvpe.onnx).

На Linux проверена сквозная конвертация RVC v1 и v2 через ONNX CPU. Настройки применяются к инференсу, включая реалтайм и извлечение признаков. Обучение генератора использует LibTorch. Под Wine отдельные графы выполнились через DirectML. Для сквозной проверки DirectML и замеров производительности нужен нативный Windows-хост.

В PyMSS для MDX23C, UVR VR, BS-RoFormer, Mel-Band-RoFormer, HyperACE, SCNet, Apollo, Bandit, HTDemucs и legacy Demucs/ConvTasNet доступен параметр `--param onnx_provider=cpu` или `--param onnx_provider=directml:0` вместе с `--device cpu`. Графы строятся из чекпойнтов. Спектральная обработка, в том числе FFT между рекуррентными блоками SCNet, выполняется на CPU. В BS-RoFormer, Mel-Band-RoFormer и HyperACE без сегментации `--param model_dtype=float16` включает FP16 для нейросети. Нормализация, вычисление углов RoPE и сегментация HyperACE используют FP32. [Результаты](bench/RESULTS.md).

В `pymss infer`, `pymss serve` и `pymss workflow run` `--device directml:0` сразу выбирает ONNX Runtime и указанный адаптер. Доступны также `--device dml:0` и `--device privateuseone:0`.

## Проверки

[Результаты сравнения с Python](bench/RESULTS.md) и [команды для повторения замеров](bench/README.md) охватывают конвертацию, реалтайм, подготовку данных, обучение и HTDemucs. Эталон для сравнительных замеров берётся из отдельной рабочей копии исходного коммита. Структура весов и медиаданные сохраняют форматы оригинального проекта.

На Linux проверены CPU и CUDA, WebUI, поток через виртуальные аудиоустройства, VST2 и VST3. CUDA-замеры выполнены на одной RTX 5060, распределение PyMSS проверено на двух CPU-репликах.

Для Windows x64 MSVC собраны `rvc.exe` и VST-библиотека с LibTorch, ONNX, FAISS и ASIO SDK 2.3.4. Под Wine VST2 и VST3 обработали PCM и сохранили настройки, API PyMSS обработал PCM на CPU и DirectML, а Windows-сборка записала индекс IVF10/Flat, который прочитал Python FAISS. Для Apple Silicon выполнена кросс-проверка типов всех целей `aarch64-apple-darwin` с `tch/doc-only`. Проверка на нативных Windows и macOS на этом стенде недоступна.

PyMSS сохраняет маршруты `/v1`. На Apple при наличии MPS PyMSS выбирает его в режиме `auto`. В CLI также доступен явный выбор `--device mps`.