#!/usr/bin/env python3
"""
Проверка эксперимента MLflow: существует ли, есть ли минимум N успешных прогонов,
залогированы ли нужные метрики и параметры.

Параметры подключения читаются из файла .env:

    MLFLOW_TRACKING_URI=http://mlflow.example.com:5000
    MLFLOW_TRACKING_USERNAME=student
    MLFLOW_TRACKING_PASSWORD=secret

Использование:
    python check_mlflow.py wb-semantic-search
    python check_mlflow.py wb-semantic-search --env-file ../.env
    python check_mlflow.py wb-semantic-search --min-runs 3
    python check_mlflow.py wb-semantic-search --metrics ndcg_at_10 mrr --params method eval_split

Код возврата: 0 - всё в порядке, 1 - есть замечания, 2 - не удалось подключиться.
"""

import argparse
import os
import sys

DEFAULT_ENV_FILE = ".env"
DEFAULT_MIN_RUNS = 2

# Что ожидаем увидеть в каждом прогоне проекта semantic search
DEFAULT_METRICS = ["ndcg_at_10", "recall_at_10", "precision_at_5", "mrr"]
DEFAULT_PARAMS = ["method", "eval_split", "catalog_size", "relevance_threshold"]

ENV_URI = "MLFLOW_TRACKING_URI"
ENV_USER = "MLFLOW_TRACKING_USERNAME"
ENV_PASSWORD = "MLFLOW_TRACKING_PASSWORD"

OK, FAIL, WARN = "  [OK]  ", "  [!]   ", "  [~]   "


def load_env_file(path):
    """
    Читает .env и кладёт значения в os.environ, не перетирая уже заданные переменные
    окружения (они имеют приоритет). Возвращает (что прочитано, текст ошибки или None).

    Сначала пробуем python-dotenv, при его отсутствии - простой встроенный парсер,
    чтобы скрипт не требовал лишних зависимостей.
    """
    if not os.path.isfile(path):
        return {}, "файл {} не найден".format(path)

    loaded = {}
    try:
        from dotenv import dotenv_values
        loaded = {k: v for k, v in dotenv_values(path).items() if v is not None}
    except ImportError:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                loaded[key] = value.strip().strip('"').strip("'")

    for key, value in loaded.items():
        os.environ.setdefault(key, value)

    return loaded, None


def main():
    parser = argparse.ArgumentParser(
        description="Проверка эксперимента MLflow на наличие прогонов, метрик и параметров."
    )
    parser.add_argument("experiment", help="название эксперимента в MLflow")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE,
                        help="путь к .env с параметрами подключения (по умолчанию {})".format(DEFAULT_ENV_FILE))
    parser.add_argument("--min-runs", type=int, default=DEFAULT_MIN_RUNS,
                        help="минимальное число успешных прогонов (по умолчанию {})".format(DEFAULT_MIN_RUNS))
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS,
                        help="обязательные метрики")
    parser.add_argument("--params", nargs="*", default=DEFAULT_PARAMS,
                        help="обязательные параметры")
    args = parser.parse_args()

    # --- 0. Подключение из .env ---
    loaded, env_error = load_env_file(args.env_file)
    if env_error:
        print("[!] {}".format(env_error))
        print("    Создайте {} с параметрами подключения:".format(args.env_file))
        print("      {}=http://localhost:5000".format(ENV_URI))
        print("      {}=...".format(ENV_USER))
        print("      {}=...".format(ENV_PASSWORD))
        return 2

    tracking_uri = os.environ.get(ENV_URI)
    username = os.environ.get(ENV_USER)
    password = os.environ.get(ENV_PASSWORD)

    if not tracking_uri:
        print("[!] В {} не задан {}.".format(args.env_file, ENV_URI))
        print("    Прочитанные переменные: {}".format(", ".join(loaded) or "ни одной"))
        return 2

    # Логин и пароль обязательны для http(s); для локального файла или sqlite они не нужны
    needs_auth = tracking_uri.startswith(("http://", "https://"))
    if needs_auth:
        missing_creds = [n for n, v in [(ENV_USER, username), (ENV_PASSWORD, password)] if not v]
        if missing_creds:
            print("[!] Для подключения по HTTP нужны учётные данные, в {} не задано: {}".format(
                args.env_file, ", ".join(missing_creds)))
            return 2

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("[!] MLflow не установлен. Установите: pip install mlflow python-dotenv")
        return 2

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    auth_note = "как {}".format(username) if (needs_auth and username) else "без авторизации"
    print("Tracking URI : {} ({})".format(tracking_uri, auth_note))
    print("Эксперимент  : {}".format(args.experiment))
    print("-" * 62)

    problems = []

    # --- 1. Эксперимент существует ---
    try:
        experiment = client.get_experiment_by_name(args.experiment)
    except Exception as e:
        print("[!] Не удалось обратиться к MLflow: {}".format(e))
        print("    Проверьте {} в {}, доступность сервера и корректность логина/пароля.".format(
            ENV_URI, args.env_file))
        return 2

    if experiment is None:
        print("[!] Эксперимент '{}' не найден.".format(args.experiment))
        try:
            available = [e.name for e in client.search_experiments()]
            print("    Доступные эксперименты: " + (", ".join(available) if available else "ни одного"))
        except Exception:
            pass
        return 1

    print("{}эксперимент найден (id={})".format(OK, experiment.experiment_id))

    # --- 2. Прогоны ---
    runs = client.search_runs([experiment.experiment_id], max_results=1000)
    finished = [r for r in runs if r.info.status == "FINISHED"]

    if len(finished) < args.min_runs:
        detail = ""
        if len(runs) != len(finished):
            detail = " (всего прогонов {}, из них незавершённых {})".format(len(runs), len(runs) - len(finished))
        problems.append("нужно минимум {} успешных прогонов, найдено {}{}".format(
            args.min_runs, len(finished), detail))
        print("{}успешных прогонов: {} (нужно {})".format(FAIL, len(finished), args.min_runs))
    else:
        print("{}успешных прогонов: {} (нужно {})".format(OK, len(finished), args.min_runs))

    if not finished:
        report(problems)
        return 1

    # --- 3. Метрики и параметры по каждому прогону ---
    print("-" * 62)
    for run in finished:
        name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
        missing_metrics = [m for m in args.metrics if m not in run.data.metrics]
        missing_params = [p for p in args.params if p not in run.data.params]

        if not missing_metrics and not missing_params:
            print("{}{}".format(OK, name))
            continue

        print("{}{}".format(FAIL, name))
        if missing_metrics:
            print("        нет метрик: {}".format(", ".join(missing_metrics)))
            problems.append("прогон '{}': не залогированы метрики {}".format(name, ", ".join(missing_metrics)))
        if missing_params:
            print("        нет параметров: {}".format(", ".join(missing_params)))
            problems.append("прогон '{}': не залогированы параметры {}".format(name, ", ".join(missing_params)))

    # --- 4. Типичная ошибка: '@' в именах метрик ---
    bad_names = sorted({m for r in finished for m in r.data.metrics if "@" in m})
    if bad_names:
        print("{}метрики с символом '@': {}".format(WARN, ", ".join(bad_names)))
        problems.append("MLflow не разрешает '@' в именах метрик - переименуйте, например ndcg@10 -> ndcg_at_10")

    # --- 5. Подсказка: сравнивать имеет смысл разные конфигурации ---
    methods = sorted({r.data.params.get("method") for r in finished if r.data.params.get("method")})
    if len(finished) >= args.min_runs and len(methods) < 2:
        print("{}все прогоны имеют одинаковый method: {}".format(WARN, ", ".join(methods) or "не задан"))
        problems.append("прогоны не различаются по параметру 'method' - сравнивать нечего")

    report(problems)
    return 1 if problems else 0


def report(problems):
    print("=" * 62)
    if problems:
        print("НАЙДЕНО ЗАМЕЧАНИЙ: {}\n".format(len(problems)))
        for i, p in enumerate(problems, 1):
            print("{}. {}".format(i, p))
    else:
        print("ВСЁ В ПОРЯДКЕ: эксперимент, прогоны, метрики и параметры на месте.")


if __name__ == "__main__":
    sys.exit(main())
