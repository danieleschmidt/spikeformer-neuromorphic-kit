"""Internationalization (i18n) support for SpikeFormer neuromorphic toolkit."""

import os
import json
import logging
from typing import Dict, Optional, Any, List, Union
from enum import Enum
from dataclasses import dataclass
import gettext
from pathlib import Path
import threading
from functools import lru_cache


class SupportedLocale(Enum):
    """Supported locales for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"


@dataclass
class LocalizationConfig:
    """Configuration for localization."""
    default_locale: str = "en"
    fallback_locale: str = "en"
    auto_detect: bool = True
    cache_translations: bool = True
    translation_domain: str = "spikeformer"
    locale_dir: str = "locales"


class TranslationManager:
    """Manages translations and locale-specific formatting."""
    
    def __init__(self, config: LocalizationConfig = None):
        self.config = config or LocalizationConfig()
        self.current_locale = self.config.default_locale
        self.translations: Dict[str, Dict[str, str]] = {}
        self.formatters: Dict[str, Any] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Initialize locale directory
        self.locale_dir = Path(__file__).parent / self.config.locale_dir
        self.locale_dir.mkdir(exist_ok=True)
        
        # Load translations
        self._initialize_translations()
        
        # Auto-detect locale if enabled
        if self.config.auto_detect:
            self._auto_detect_locale()
    
    def _initialize_translations(self):
        """Initialize default translations."""
        # Create default translation files if they don't exist
        default_translations = {
            "en": {
                # Error messages
                "error.model_not_found": "Model not found: {model_name}",
                "error.invalid_input": "Invalid input provided: {details}",
                "error.conversion_failed": "Model conversion failed: {reason}",
                "error.hardware_incompatible": "Hardware not compatible: {hardware_type}",
                "error.insufficient_memory": "Insufficient memory for operation",
                "error.timeout": "Operation timed out after {seconds} seconds",
                
                # Success messages
                "success.model_loaded": "Model loaded successfully: {model_name}",
                "success.conversion_complete": "Conversion completed in {time:.2f} seconds",
                "success.training_complete": "Training completed with accuracy: {accuracy:.2%}",
                "success.benchmark_complete": "Benchmark completed: {metrics}",
                
                # Info messages
                "info.starting_training": "Starting training with {epochs} epochs",
                "info.loading_dataset": "Loading dataset: {dataset_name}",
                "info.initializing_hardware": "Initializing {hardware_type} hardware",
                "info.saving_checkpoint": "Saving checkpoint at epoch {epoch}",
                
                # UI labels
                "ui.model_name": "Model Name",
                "ui.accuracy": "Accuracy",
                "ui.energy_consumption": "Energy Consumption",
                "ui.inference_time": "Inference Time",
                "ui.batch_size": "Batch Size",
                "ui.learning_rate": "Learning Rate",
                "ui.epochs": "Epochs",
                "ui.hardware_backend": "Hardware Backend",
                
                # Units and formatting
                "unit.milliseconds": "ms",
                "unit.microseconds": "μs",
                "unit.millijoules": "mJ",
                "unit.microjoules": "μJ",
                "unit.percentage": "%",
                "unit.megabytes": "MB",
                "unit.gigabytes": "GB",
                
                # Hardware types
                "hardware.cpu": "CPU",
                "hardware.gpu": "GPU", 
                "hardware.loihi2": "Intel Loihi 2",
                "hardware.spinnaker": "SpiNNaker",
                "hardware.brainscales": "BrainScaleS",
                
                # Model types
                "model.spiking_transformer": "Spiking Transformer",
                "model.spiking_vit": "Spiking Vision Transformer",
                "model.spiking_bert": "Spiking BERT",
                "model.multi_modal": "Multi-Modal Spiking Model",
                
                # Progress messages
                "progress.initializing": "Initializing...",
                "progress.loading": "Loading...",
                "progress.processing": "Processing...",
                "progress.saving": "Saving...",
                "progress.complete": "Complete",
                
                # Warnings
                "warning.low_memory": "Warning: Low memory available",
                "warning.deprecated": "Warning: {feature} is deprecated",
                "warning.experimental": "Warning: {feature} is experimental",
            },
            
            "es": {
                # Error messages
                "error.model_not_found": "Modelo no encontrado: {model_name}",
                "error.invalid_input": "Entrada inválida proporcionada: {details}",
                "error.conversion_failed": "Falló la conversión del modelo: {reason}",
                "error.hardware_incompatible": "Hardware no compatible: {hardware_type}",
                "error.insufficient_memory": "Memoria insuficiente para la operación",
                "error.timeout": "Operación agotó el tiempo tras {seconds} segundos",
                
                # Success messages
                "success.model_loaded": "Modelo cargado exitosamente: {model_name}",
                "success.conversion_complete": "Conversión completada en {time:.2f} segundos",
                "success.training_complete": "Entrenamiento completado con precisión: {accuracy:.2%}",
                "success.benchmark_complete": "Benchmark completado: {metrics}",
                
                # UI labels
                "ui.model_name": "Nombre del Modelo",
                "ui.accuracy": "Precisión",
                "ui.energy_consumption": "Consumo de Energía",
                "ui.inference_time": "Tiempo de Inferencia",
                "ui.batch_size": "Tamaño de Lote",
                "ui.learning_rate": "Tasa de Aprendizaje",
                "ui.epochs": "Épocas",
                "ui.hardware_backend": "Backend de Hardware",
                
                # Hardware types
                "hardware.cpu": "CPU",
                "hardware.gpu": "GPU",
                "hardware.loihi2": "Intel Loihi 2",
                "hardware.spinnaker": "SpiNNaker",
                "hardware.brainscales": "BrainScaleS",
            },
            
            "fr": {
                # Error messages  
                "error.model_not_found": "Modèle introuvable: {model_name}",
                "error.invalid_input": "Entrée invalide fournie: {details}",
                "error.conversion_failed": "Échec de la conversion du modèle: {reason}",
                "error.hardware_incompatible": "Matériel non compatible: {hardware_type}",
                "error.insufficient_memory": "Mémoire insuffisante pour l'opération",
                "error.timeout": "Opération expirée après {seconds} secondes",
                
                # UI labels
                "ui.model_name": "Nom du Modèle",
                "ui.accuracy": "Précision",
                "ui.energy_consumption": "Consommation d'Énergie",
                "ui.inference_time": "Temps d'Inférence",
                "ui.batch_size": "Taille de Lot",
                "ui.learning_rate": "Taux d'Apprentissage",
                "ui.epochs": "Époques",
                "ui.hardware_backend": "Backend Matériel",
            },
            
            "de": {
                # Error messages
                "error.model_not_found": "Modell nicht gefunden: {model_name}",
                "error.invalid_input": "Ungültige Eingabe bereitgestellt: {details}",
                "error.conversion_failed": "Modellkonvertierung fehlgeschlagen: {reason}",
                "error.hardware_incompatible": "Hardware nicht kompatibel: {hardware_type}",
                "error.insufficient_memory": "Unzureichender Speicher für Operation",
                "error.timeout": "Operation nach {seconds} Sekunden abgebrochen",
                
                # UI labels
                "ui.model_name": "Modellname",
                "ui.accuracy": "Genauigkeit", 
                "ui.energy_consumption": "Energieverbrauch",
                "ui.inference_time": "Inferenzzeit",
                "ui.batch_size": "Stapelgröße",
                "ui.learning_rate": "Lernrate",
                "ui.epochs": "Epochen",
                "ui.hardware_backend": "Hardware-Backend",
            },
            
            "ja": {
                # Error messages
                "error.model_not_found": "モデルが見つかりません: {model_name}",
                "error.invalid_input": "無効な入力が提供されました: {details}",
                "error.conversion_failed": "モデル変換が失敗しました: {reason}",
                "error.hardware_incompatible": "ハードウェアが非互換です: {hardware_type}",
                "error.insufficient_memory": "操作のためのメモリが不足しています",
                "error.timeout": "操作が{seconds}秒後にタイムアウトしました",
                
                # UI labels
                "ui.model_name": "モデル名",
                "ui.accuracy": "精度",
                "ui.energy_consumption": "エネルギー消費",
                "ui.inference_time": "推論時間",
                "ui.batch_size": "バッチサイズ",
                "ui.learning_rate": "学習率",
                "ui.epochs": "エポック",
                "ui.hardware_backend": "ハードウェアバックエンド",
            },
            
            "zh-CN": {
                # Error messages
                "error.model_not_found": "未找到模型: {model_name}",
                "error.invalid_input": "提供了无效输入: {details}",
                "error.conversion_failed": "模型转换失败: {reason}",
                "error.hardware_incompatible": "硬件不兼容: {hardware_type}",
                "error.insufficient_memory": "操作内存不足",
                "error.timeout": "操作在{seconds}秒后超时",
                
                # UI labels
                "ui.model_name": "模型名称",
                "ui.accuracy": "准确率",
                "ui.energy_consumption": "能耗",
                "ui.inference_time": "推理时间",
                "ui.batch_size": "批大小",
                "ui.learning_rate": "学习率",
                "ui.epochs": "轮次",
                "ui.hardware_backend": "硬件后端",
            }
        }
        
        # Save translation files
        for locale, translations in default_translations.items():
            self._save_translation_file(locale, translations)
        
        # Load all translations
        self._load_all_translations()
    
    def _save_translation_file(self, locale: str, translations: Dict[str, str]):
        """Save translations to file."""
        locale_file = self.locale_dir / f"{locale}.json"
        try:
            with open(locale_file, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save translations for {locale}: {e}")
    
    def _load_all_translations(self):
        """Load all available translations."""
        self.translations.clear()
        
        for locale_file in self.locale_dir.glob("*.json"):
            locale = locale_file.stem
            try:
                with open(locale_file, 'r', encoding='utf-8') as f:
                    translations = json.load(f)
                    self.translations[locale] = translations
            except Exception as e:
                self.logger.warning(f"Failed to load translations for {locale}: {e}")
    
    def _auto_detect_locale(self):
        """Auto-detect system locale."""
        try:
            import locale as sys_locale
            system_locale = sys_locale.getdefaultlocale()[0]
            
            if system_locale:
                # Extract language code (e.g., 'en_US' -> 'en')
                lang_code = system_locale.split('_')[0]
                
                # Check if we support this locale
                if lang_code in [loc.value for loc in SupportedLocale]:
                    self.current_locale = lang_code
                    self.logger.info(f"Auto-detected locale: {lang_code}")
                
        except Exception as e:
            self.logger.warning(f"Failed to auto-detect locale: {e}")
    
    def set_locale(self, locale: str):
        """Set current locale."""
        with self.lock:
            if locale in self.translations:
                self.current_locale = locale
                self.logger.info(f"Locale changed to: {locale}")
            else:
                self.logger.warning(f"Locale not supported: {locale}")
                raise ValueError(f"Unsupported locale: {locale}")
    
    def get_locale(self) -> str:
        """Get current locale."""
        return self.current_locale
    
    @lru_cache(maxsize=1000)
    def translate(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """Translate a key to the current or specified locale."""
        target_locale = locale or self.current_locale
        
        with self.lock:
            # Try target locale
            if target_locale in self.translations:
                translation = self.translations[target_locale].get(key)
                if translation:
                    try:
                        return translation.format(**kwargs) if kwargs else translation
                    except KeyError as e:
                        self.logger.warning(f"Missing format parameter {e} for key '{key}'")
                        return translation
            
            # Fallback to default locale
            if (self.config.fallback_locale != target_locale and 
                self.config.fallback_locale in self.translations):
                
                translation = self.translations[self.config.fallback_locale].get(key)
                if translation:
                    try:
                        return translation.format(**kwargs) if kwargs else translation
                    except KeyError:
                        return translation
            
            # Return key if no translation found
            self.logger.warning(f"No translation found for key: {key}")
            return key
    
    def t(self, key: str, **kwargs) -> str:
        """Short alias for translate."""
        return self.translate(key, **kwargs)
    
    def get_available_locales(self) -> List[str]:
        """Get list of available locales."""
        return list(self.translations.keys())
    
    def format_number(self, number: Union[int, float], locale: Optional[str] = None) -> str:
        """Format number according to locale conventions."""
        target_locale = locale or self.current_locale
        
        # Simple locale-specific number formatting
        if target_locale in ['en']:
            return f"{number:,}"
        elif target_locale in ['de', 'es', 'fr']:
            # European style: use . for thousands separator, , for decimals
            if isinstance(number, float):
                return f"{number:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
            else:
                return f"{number:,}".replace(',', '.')
        else:
            return str(number)
    
    def format_currency(self, amount: float, currency: str = "USD", 
                       locale: Optional[str] = None) -> str:
        """Format currency according to locale conventions."""
        target_locale = locale or self.current_locale
        formatted_amount = self.format_number(amount, target_locale)
        
        # Simple currency formatting
        currency_symbols = {
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'CNY': '¥',
        }
        
        symbol = currency_symbols.get(currency, currency)
        
        if target_locale in ['en']:
            return f"{symbol}{formatted_amount}"
        elif target_locale in ['de', 'es', 'fr']:
            return f"{formatted_amount} {symbol}"
        else:
            return f"{symbol} {formatted_amount}"
    
    def format_percentage(self, value: float, decimals: int = 1,
                         locale: Optional[str] = None) -> str:
        """Format percentage according to locale conventions."""
        target_locale = locale or self.current_locale
        percentage = value * 100
        
        if target_locale in ['en']:
            return f"{percentage:.{decimals}f}%"
        elif target_locale in ['de', 'fr']:
            return f"{percentage:.{decimals}f}".replace('.', ',') + " %"
        else:
            return f"{percentage:.{decimals}f}%"
    
    def format_energy(self, energy_uj: float, locale: Optional[str] = None) -> str:
        """Format energy consumption with appropriate units."""
        if energy_uj >= 1000000:  # >= 1 J
            return f"{energy_uj/1000000:.2f} J"
        elif energy_uj >= 1000:  # >= 1 mJ
            return f"{energy_uj/1000:.2f} {self.t('unit.millijoules')}"
        else:
            return f"{energy_uj:.2f} {self.t('unit.microjoules')}"
    
    def format_time(self, time_seconds: float, locale: Optional[str] = None) -> str:
        """Format time with appropriate units."""
        if time_seconds >= 1.0:
            return f"{time_seconds:.2f} s"
        elif time_seconds >= 0.001:  # >= 1 ms
            return f"{time_seconds*1000:.2f} {self.t('unit.milliseconds')}"
        else:
            return f"{time_seconds*1000000:.2f} {self.t('unit.microseconds')}"
    
    def add_custom_translations(self, locale: str, translations: Dict[str, str]):
        """Add custom translations for a locale."""
        with self.lock:
            if locale not in self.translations:
                self.translations[locale] = {}
            
            self.translations[locale].update(translations)
            
            # Save to file
            self._save_translation_file(locale, self.translations[locale])
    
    def reload_translations(self):
        """Reload translations from files."""
        with self.lock:
            self._load_all_translations()
            # Clear cache
            self.translate.cache_clear()


# Global translation manager instance
_translation_manager = None
_manager_lock = threading.Lock()


def get_translation_manager() -> TranslationManager:
    """Get global translation manager instance."""
    global _translation_manager
    
    if _translation_manager is None:
        with _manager_lock:
            if _translation_manager is None:
                _translation_manager = TranslationManager()
    
    return _translation_manager


def set_locale(locale: str):
    """Set global locale."""
    get_translation_manager().set_locale(locale)


def get_locale() -> str:
    """Get current global locale."""
    return get_translation_manager().get_locale()


def t(key: str, **kwargs) -> str:
    """Global translation function."""
    return get_translation_manager().translate(key, **kwargs)


def format_energy(energy_uj: float) -> str:
    """Global energy formatting function."""
    return get_translation_manager().format_energy(energy_uj)


def format_time(time_seconds: float) -> str:
    """Global time formatting function."""
    return get_translation_manager().format_time(time_seconds)


def format_percentage(value: float, decimals: int = 1) -> str:
    """Global percentage formatting function."""
    return get_translation_manager().format_percentage(value, decimals)


def get_available_locales() -> List[str]:
    """Get available locales."""
    return get_translation_manager().get_available_locales()


class LocalizedError(Exception):
    """Exception with localized error messages."""
    
    def __init__(self, message_key: str, **kwargs):
        self.message_key = message_key
        self.kwargs = kwargs
        localized_message = t(message_key, **kwargs)
        super().__init__(localized_message)
    
    def get_localized_message(self, locale: Optional[str] = None) -> str:
        """Get error message in specified locale."""
        return get_translation_manager().translate(self.message_key, locale, **self.kwargs)