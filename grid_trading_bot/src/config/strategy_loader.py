"""
Strategy Configuration Loader.

從 settings.yaml 加載策略參數，支援環境變數覆蓋。
實現「單一來源」設計：只需修改 settings.yaml，所有相關文件自動同步參數。

Usage:
    from src.config.strategy_loader import load_strategy_config

    # 從 YAML 加載策略參數
    params = load_strategy_config("grid_futures")  # 對應 bot_id: grid_futures_*

    # 支援環境變數覆蓋
    # GRID_FUTURES_LEVERAGE=20 python run_grid_futures.py
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml

# 預設配置文件路徑
DEFAULT_SETTINGS_PATH = Path(__file__).parent.parent / "fund_manager" / "config" / "settings.yaml"


def _env_var_name(bot_type: str, param_name: str) -> str:
    """
    生成環境變數名稱。

    格式: {BOT_TYPE}_{PARAM_NAME}
    例如: grid_futures + leverage -> GRID_FUTURES_LEVERAGE

    Args:
        bot_type: Bot 類型 (e.g., "grid_futures", "bollinger")
        param_name: 參數名稱 (e.g., "leverage", "grid_count")

    Returns:
        環境變數名稱 (大寫，底線分隔)
    """
    return f"{bot_type.upper()}_{param_name.upper()}"


def _convert_value(value: str, target_type: type) -> Any:
    """
    將環境變數字串轉換為目標類型。

    Args:
        value: 環境變數字串值
        target_type: 目標類型

    Returns:
        轉換後的值
    """
    if target_type == bool:
        return value.lower() in ("true", "yes", "1", "on")
    elif target_type == int:
        return int(value)
    elif target_type == float:
        return float(value)
    else:
        return value


def _get_bot_pattern(bot_type: str) -> str:
    """
    獲取 bot_id 匹配模式。

    Args:
        bot_type: Bot 類型 (e.g., "grid_futures", "bollinger")

    Returns:
        bot_id 模式 (e.g., "grid_futures_*", "bollinger_*")
    """
    return f"{bot_type}_*"


def load_strategy_config(
    bot_type: str,
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    從 settings.yaml 加載策略參數，支援環境變數覆蓋。

    單一來源設計：
    1. 從 settings.yaml 讀取基礎參數
    2. 應用環境變數覆蓋 (格式: {BOT_TYPE}_{PARAM_NAME})

    Args:
        bot_type: Bot 類型名稱 (e.g., "grid_futures", "bollinger", "supertrend", "rsi_grid")
        settings_path: settings.yaml 路徑（可選，默認使用標準路徑）

    Returns:
        策略參數字典

    Raises:
        FileNotFoundError: 如果找不到 settings.yaml
        ValueError: 如果找不到對應的 bot 配置

    Example:
        >>> params = load_strategy_config("grid_futures")
        >>> print(params["leverage"])  # 從 YAML 讀取
        10

        >>> # 環境變數覆蓋
        >>> # export GRID_FUTURES_LEVERAGE=20
        >>> params = load_strategy_config("grid_futures")
        >>> print(params["leverage"])
        20
    """
    # 確定配置文件路徑
    if settings_path is None:
        settings_path = DEFAULT_SETTINGS_PATH
    else:
        settings_path = Path(settings_path)

    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    # 讀取 YAML 配置
    with open(settings_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 查找對應的 bot 配置
    bots = config.get("bots", [])
    bot_pattern = _get_bot_pattern(bot_type)

    strategy_params = None
    for bot in bots:
        if bot.get("bot_id") == bot_pattern:
            strategy_params = bot.get("strategy_params", {})
            break

    if strategy_params is None:
        raise ValueError(
            f"Bot configuration not found for type '{bot_type}' "
            f"(looking for bot_id: '{bot_pattern}')"
        )

    # 複製參數以避免修改原始配置
    params = dict(strategy_params)

    # 應用環境變數覆蓋
    for param_name, default_value in strategy_params.items():
        env_var = _env_var_name(bot_type, param_name)
        env_value = os.getenv(env_var)

        if env_value is not None:
            # 根據原始值類型轉換環境變數
            original_type = type(default_value)
            params[param_name] = _convert_value(env_value, original_type)

    return params


def validate_config_consistency(
    bot_type: str,
    config_obj,
    settings_path: str | Path | None = None,
) -> list[str]:
    """
    比對 dataclass 實際值與 settings.yaml 值，回報差異。

    Args:
        bot_type: Bot 類型名稱 (e.g., "grid_futures", "bollinger", "supertrend")
        config_obj: 已實例化的 config 物件 (dataclass)
        settings_path: settings.yaml 路徑（可選）

    Returns:
        差異清單，每筆格式: "{param}: yaml={yaml_val} actual={actual_val}"
    """
    from decimal import Decimal

    try:
        yaml_params = load_strategy_config(bot_type, settings_path)
    except (FileNotFoundError, ValueError):
        return [f"Cannot load settings.yaml for bot_type='{bot_type}'"]

    warnings = []
    for param_name, yaml_val in yaml_params.items():
        if not hasattr(config_obj, param_name):
            continue
        actual_val = getattr(config_obj, param_name)

        # Normalize for comparison: convert both to same type
        try:
            if isinstance(actual_val, Decimal):
                yaml_comparable = Decimal(str(yaml_val))
                actual_comparable = actual_val
            elif isinstance(actual_val, bool):
                yaml_comparable = bool(yaml_val) if not isinstance(yaml_val, bool) else yaml_val
                actual_comparable = actual_val
            elif isinstance(actual_val, int):
                yaml_comparable = int(yaml_val)
                actual_comparable = actual_val
            elif isinstance(actual_val, float):
                yaml_comparable = float(yaml_val)
                actual_comparable = actual_val
            else:
                yaml_comparable = str(yaml_val)
                actual_comparable = str(actual_val)

            if yaml_comparable != actual_comparable:
                warnings.append(
                    f"{param_name}: yaml={yaml_val} actual={actual_val}"
                )
        except (ValueError, TypeError):
            warnings.append(
                f"{param_name}: type mismatch yaml={yaml_val!r} actual={actual_val!r}"
            )

    return warnings


def get_all_bot_configs(
    settings_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """
    獲取所有 bot 的策略配置。

    Args:
        settings_path: settings.yaml 路徑（可選）

    Returns:
        字典，key 為 bot_type，value 為策略參數

    Example:
        >>> configs = get_all_bot_configs()
        >>> for bot_type, params in configs.items():
        ...     print(f"{bot_type}: leverage={params.get('leverage')}")
    """
    # 確定配置文件路徑
    if settings_path is None:
        settings_path = DEFAULT_SETTINGS_PATH
    else:
        settings_path = Path(settings_path)

    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    # 讀取 YAML 配置
    with open(settings_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 提取所有 bot 的策略參數
    result = {}
    bots = config.get("bots", [])

    for bot in bots:
        bot_id = bot.get("bot_id", "")
        # 從 bot_id 模式提取 bot_type (例如: "grid_futures_*" -> "grid_futures")
        if bot_id.endswith("_*"):
            bot_type = bot_id[:-2]  # 移除 "_*"
            strategy_params = bot.get("strategy_params", {})
            if strategy_params:
                result[bot_type] = load_strategy_config(bot_type, settings_path)

    return result


def get_bot_allocation(
    bot_type: str,
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    獲取 bot 的資金分配設定。

    Args:
        bot_type: Bot 類型名稱
        settings_path: settings.yaml 路徑（可選）

    Returns:
        資金分配設定字典 (ratio, min_capital, max_capital, priority)

    Example:
        >>> allocation = get_bot_allocation("grid_futures")
        >>> print(f"ratio={allocation['ratio']}, max={allocation['max_capital']}")
    """
    # 確定配置文件路徑
    if settings_path is None:
        settings_path = DEFAULT_SETTINGS_PATH
    else:
        settings_path = Path(settings_path)

    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    # 讀取 YAML 配置
    with open(settings_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 查找對應的 bot 配置
    bots = config.get("bots", [])
    bot_pattern = _get_bot_pattern(bot_type)

    for bot in bots:
        if bot.get("bot_id") == bot_pattern:
            return {
                "ratio": bot.get("ratio", 0.0),
                "min_capital": bot.get("min_capital", 0),
                "max_capital": bot.get("max_capital", None),
                "priority": bot.get("priority", 0),
                "status": bot.get("status", "inactive"),
            }

    raise ValueError(
        f"Bot configuration not found for type '{bot_type}' "
        f"(looking for bot_id: '{bot_pattern}')"
    )
