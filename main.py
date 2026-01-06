#!/usr/bin/env python3
"""
Tatar Text Detoxifier - Rule-Based Approach
Улучшенная версия оригинального h2.py с объектно-ориентированным дизайном,
обработкой ошибок и расширенной функциональностью.
"""

import re
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


class TatarDetoxifier:
    """
    Основной класс для детоксификации татарского текста.
    Реализует многоуровневую фильтрацию на основе словарей.
    """
    
    def __init__(self, 
                 replacements_file: str = "toxic_replacements.json",
                 substrings_file: str = "toxic_substrings.json",
                 twl_file: str = "tat_Cyrl_twl.txt",
                 lexicon_file: str = "tt_ru_lexicon.csv",
                 use_lexicon: bool = True):
        """
        Инициализация детоксификатора с загрузкой словарей.
        
        Args:
            replacements_file: JSON файл с заменой токсичных слов
            substrings_file: JSON файл с токсичными корнями
            twl_file: Текстовый файл с нежелательными словами
            lexicon_file: CSV лексикон для гибридных форм
            use_lexicon: Использовать ли внешний лексикон
        """
        self.base_dir = Path(__file__).resolve().parent
        
        # Загрузка словарей с обработкой ошибок
        self.toxic_replacements = self._load_replacements(replacements_file)
        self.toxic_substrings = self._load_substrings(substrings_file)
        self.toxic_twl = self._load_twl(twl_file)
        
        # Построение полного словаря токсичных слов
        self.toxic_dict_full = self._build_toxic_dict(lexicon_file, use_lexicon)
        self.toxic_dict_base = self._build_toxic_dict(lexicon_file, False)
        
        # Компиляция regex паттернов для поиска
        self.pattern_full = self._build_pattern(self.toxic_dict_full)
        self.pattern_base = self._build_pattern(self.toxic_dict_base)
        
        # Статистика загруженных данных
        self.stats = {
            'replacements': len(self.toxic_replacements),
            'substrings': len(self.toxic_substrings),
            'twl_words': len(self.toxic_twl),
            'full_dict': len(self.toxic_dict_full),
            'base_dict': len(self.toxic_dict_base)
        }
    
    def _load_replacements(self, filename: str) -> Dict[str, str]:
        """Загрузка словаря замен из JSON файла."""
        filepath = self.base_dir / filename
        if not filepath.exists():
            print(f"Внимание: файл {filename} не найден, используется пустой словарь замен")
            return {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Ошибка чтения JSON файла {filename}: {e}")
            return {}
    
    def _load_substrings(self, filename: str) -> List[str]:
        """Загрузка списка токсичных корней из JSON файла."""
        filepath = self.base_dir / filename
        if not filepath.exists():
            print(f"Внимание: файл {filename} не найден, используется пустой список корней")
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Ошибка чтения JSON файла {filename}: {e}")
            return []
    
    def _load_twl(self, filename: str) -> List[str]:
        """Загрузка списка нежелательных слов из текстового файла."""
        filepath = self.base_dir / filename
        if not filepath.exists():
            print(f"Внимание: файл {filename} не найден")
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Ошибка чтения файла {filename}: {e}")
            return []
    
    def _is_toxic_by_substring(self, word: str) -> bool:
        """
        Проверка содержит ли слово токсичные корни.
        
        Args:
            word: Слово для проверки
            
        Returns:
            True если содержит токсичный корень
        """
        word_lower = word.lower()
        return any(sub in word_lower for sub in self.toxic_substrings)
    
    def _build_toxic_dict(self, lexicon_file: str, use_lexicon: bool) -> Dict[str, str]:
        """
        Построение полного словаря токсичных слов из всех источников.
        
        Args:
            lexicon_file: Путь к CSV лексикону
            use_lexicon: Включать ли слова из лексикона
            
        Returns:
            Словарь {токсичное_слово: замена}
        """
        toxic_dict = {}
        
        # 1. Добавляем замены из основного словаря
        for word, replacement in self.toxic_replacements.items():
            toxic_dict[word.lower()] = replacement
        
        # 2. Добавляем слова для полного удаления из TWL
        for word in self.toxic_twl:
            toxic_dict[word.lower()] = ""
        
        # 3. Добавляем слова из лексикона (если нужно)
        if use_lexicon:
            self._add_lexicon_words(toxic_dict, lexicon_file)
        
        return toxic_dict
    
    def _add_lexicon_words(self, toxic_dict: Dict[str, str], lexicon_file: str) -> None:
        """
        Добавление токсичных слов из CSV лексикона.
        
        Args:
            toxic_dict: Словарь для дополнения
            lexicon_file: Путь к CSV файлу
        """
        lexicon_path = self.base_dir / lexicon_file
        if not lexicon_path.exists():
            print(f"Внимание: лексикон {lexicon_file} не найден")
            return
        
        try:
            # Пробуем разные кодировки
            try:
                lex_df = pd.read_csv(lexicon_path, encoding='utf-8')
            except UnicodeDecodeError:
                lex_df = pd.read_csv(lexicon_path, encoding='cp1251')
            
            # Ищем колонку с текстом
            text_col = None
            for col in ['text', 'word', 'token', 'phrase', 'слово']:
                if col in lex_df.columns:
                    text_col = col
                    break
            
            if text_col:
                for word in lex_df[text_col].astype(str):
                    word_clean = word.strip()
                    if word_clean and self._is_toxic_by_substring(word_clean):
                        toxic_dict[word_clean.lower()] = ""
            else:
                print("Внимание: не найдена текстовая колонка в лексиконе")
                
        except Exception as e:
            print(f"Ошибка загрузки лексикона {lexicon_file}: {e}")
    
    def _build_pattern(self, toxic_dict: Dict[str, str]) -> Optional[re.Pattern]:
        """
        Создание regex паттерна для поиска токсичных слов.
        
        Args:
            toxic_dict: Словарь токсичных слов
            
        Returns:
            Скомпилированный regex паттерн или None
        """
        if not toxic_dict:
            return None
        
        # Сортируем по длине для корректного поиска (длинные слова первыми)
        sorted_words = sorted(toxic_dict.keys(), key=len, reverse=True)
        pattern_str = r'\b(' + '|'.join(re.escape(w) for w in sorted_words) + r')\b'
        
        return re.compile(pattern_str, flags=re.IGNORECASE | re.UNICODE)
    
    def _apply_case_preservation(self, original: str, replacement: str) -> str:
        """
        Сохранение регистра оригинального слова в замене.
        
        Args:
            original: Оригинальное слово
            replacement: Слово для замены
            
        Returns:
            Замена с сохраненным регистром
        """
        if not replacement:
            return ""
        
        if original.isupper():
            return replacement.upper()
        elif original and original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        else:
            return replacement
    
    def _clean_punctuation(self, text: str) -> str:
        """
        Очистка пунктуации и лишних пробелов.
        
        Args:
            text: Текст для очистки
            
        Returns:
            Очищенный текст
        """
        # Убираем пробелы перед знаками препинания
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        # Убираем лишние пробелы
        text = re.sub(r'\s{2,}', ' ', text)
        # Убираем пробелы в начале и конце
        return text.strip()
    
    def detoxify(self, text: str) -> str:
        """
        Основная функция детоксификации текста.
        
        Args:
            text: Исходный текст на татарском языке
            
        Returns:
            Очищенный текст
        """
        if not isinstance(text, str) or not text.strip():
            return text
        
        original_text = text
        original_tokens = text.split()
        original_length = len(original_tokens)
        
        # Шаг 1: Замена токсичных слов (полный словарь)
        if self.pattern_full:
            text = self._apply_dict_replacements(text, self.toxic_dict_full, self.pattern_full)
        
        # Шаг 2: Удаление остатков по подстрокам
        text = self._remove_residual_toxicity(text)
        
        # Шаг 3: Fallback если текст стал слишком коротким
        cleaned_tokens = text.split()
        if original_length > 0 and len(cleaned_tokens) < max(3, int(original_length * 0.6)):
            # Используем базовый словарь (без лексикона)
            if self.pattern_base:
                text = self._apply_dict_replacements(original_text, self.toxic_dict_base, self.pattern_base)
                text = self._remove_residual_toxicity(text)
        
        return text
    
    def detoxify_with_log(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Детоксификация с возвратом лога изменений.
        
        Args:
            text: Исходный текст
            
        Returns:
            Кортеж (очищенный_текст, список_изменений)
        """
        changes = []
        
        if not isinstance(text, str) or not text.strip():
            return text, changes
        
        def log_replacement(match):
            original_word = match.group(0)
            replacement = self.toxic_dict_full.get(original_word.lower(), "")
            
            changes.append({
                'original': original_word,
                'replacement': replacement,
                'position': match.start()
            })
            
            return self._apply_case_preservation(original_word, replacement)
        
        # Применяем замены с логированием
        if self.pattern_full:
            text = self.pattern_full.sub(log_replacement, text)
        
        # Удаляем остатки по подстрокам
        text = self._remove_residual_toxicity(text)
        
        return text, changes
    
    def _apply_dict_replacements(self, text: str, toxic_dict: Dict[str, str], pattern: re.Pattern) -> str:
        """
        Применение замен из словаря к тексту.
        
        Args:
            text: Исходный текст
            toxic_dict: Словарь замен
            pattern: Regex паттерн
            
        Returns:
            Текст с примененными заменами
        """
        def replacement_func(match):
            original_word = match.group(0)
            replacement = toxic_dict.get(original_word.lower(), "")
            return self._apply_case_preservation(original_word, replacement)
        
        return pattern.sub(replacement_func, text)
    
    def _remove_residual_toxicity(self, text: str) -> str:
        """
        Удаление остаточных токсичных элементов по подстрокам.
        
        Args:
            text: Текст для очистки
            
        Returns:
            Очищенный текст
        """
        tokens = text.split()
        cleaned_tokens = []
        
        for token in tokens:
            # Сохраняем пунктуацию отдельно
            if token in ['.', ',', '!', '?', ';', ':']:
                cleaned_tokens.append(token)
                continue
            
            # Проверяем токсичность
            if not self._is_toxic_by_substring(token):
                cleaned_tokens.append(token)
        
        return self._clean_punctuation(' '.join(cleaned_tokens))
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Получение статистики по загруженным словарям.
        
        Returns:
            Словарь со статистикой
        """
        return self.stats.copy()
    
    def get_toxic_words(self) -> List[str]:
        """
        Получение списка всех токсичных слов в словаре.
        
        Returns:
            Список токсичных слов
        """
        return list(self.toxic_dict_full.keys())


# Глобальный экземпляр для удобства использования
_detoxifier_instance = None

def get_detoxifier() -> TatarDetoxifier:
    """
    Получение глобального экземпляра детоксификатора.
    
    Returns:
        Экземпляр TatarDetoxifier
    """
    global _detoxifier_instance
    if _detoxifier_instance is None:
        _detoxifier_instance = TatarDetoxifier()
    return _detoxifier_instance

def detoxify(text: str) -> str:
    """
    Упрощенная функция для быстрой детоксификации текста.
    
    Args:
        text: Текст для очистки
        
    Returns:
        Очищенный текст
    """
    return get_detoxifier().detoxify(text)


def process_tsv_file(input_path: str, output_path: str, text_column: str = "tat_toxic") -> None:
    """
    Обработка TSV файла с текстами.
    
    Args:
        input_path: Путь к входному TSV файлу
        output_path: Путь для сохранения результата
        text_column: Название колонки с текстом
    """
    try:
        # Чтение файла
        df = pd.read_csv(input_path, sep='\t', dtype=str, keep_default_na=False)
        
        if text_column not in df.columns:
            available_columns = list(df.columns)
            raise ValueError(f"Колонка '{text_column}' не найдена в файле. Доступные колонки: {available_columns}")
        
        print(f"Обработка {len(df)} строк...")
        
        # Применение детоксификации
        detoxifier = get_detoxifier()
        df["tat_detox"] = df[text_column].apply(detoxifier.detoxify)
        
        # Сохранение результата
        df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        print(f"Результат сохранен в {output_path}")
        
        # Вывод примеров обработки
        if len(df) > 0:
            print("\nПримеры обработки:")
            for i in range(min(3, len(df))):
                original = df.iloc[i][text_column]
                cleaned = df.iloc[i]["tat_detox"]
                print(f"\n{i+1}. ДО:   {original}")
                print(f"   ПОСЛЕ: {cleaned}")
                
    except Exception as e:
        print(f"Ошибка обработки файла: {e}")
        raise


def run_tests():
    """Запуск базовых тестов для проверки работоспособности."""
    print("Запуск тестов...")
    
    detoxifier = TatarDetoxifier(use_lexicon=False)
    
    # Тестовые случаи
    test_cases = [
        ("син дурак", "син ялгышасың"),
        ("ахмак идея", "ялгышасың идея"),
        ("", ""),
        ("простой текст", "простой текст"),
    ]
    
    all_passed = True
    for input_text, expected in test_cases:
        result = detoxifier.detoxify(input_text)
        if result == expected:
            print(f"✓ Тест пройден: '{input_text}' -> '{result}'")
        else:
            print(f"✗ Тест не пройден: '{input_text}' -> '{result}', ожидалось '{expected}'")
            all_passed = False
    
    # Тест с логированием
    text = "син дурак и ахмак"
    cleaned, changes = detoxifier.detoxify_with_log(text)
    print(f"\nТест логирования: '{text}' -> '{cleaned}'")
    print(f"Количество изменений: {len(changes)}")
    
    if all_passed:
        print("\nВсе базовые тесты пройдены успешно")
    else:
        print("\nНекоторые тесты не пройдены")
    
    return all_passed


def main():
    """Основная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description="Детоксификатор татарского текста (rule-based подход)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python tatar_detox.py --text "син нинди дурак кеше"
  python tatar_detox.py -i input.tsv -o output.tsv
  python tatar_detox.py -i input.tsv -o output.tsv --column text
  python tatar_detox.py --test  # запуск тестов
        """
    )
    
    parser.add_argument(
        "--text", 
        type=str,
        help="Текст для детоксификации"
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Путь к входному TSV файлу"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Путь для сохранения результата (только для файлового режима)"
    )
    
    parser.add_argument(
        "--column",
        type=str,
        default="tat_toxic",
        help="Название колонки с текстом (по умолчанию: 'tat_toxic')"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Запуск тестов"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Показать статистику загруженных словарей"
    )
    
    args = parser.parse_args()
    
    # Режим запуска тестов
    if args.test:
        run_tests()
        return
    
    # Показать статистику
    if args.stats:
        detoxifier = get_detoxifier()
        stats = detoxifier.get_statistics()
        print("Статистика загруженных словарей:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    # Режим обработки единичного текста
    if args.text:
        result = detoxify(args.text)
        print(f"\nРезультат:")
        print(f"   Исходный: {args.text}")
        print(f"   Очищенный: {result}")
        return
    
    # Режим обработки файла
    if args.input:
        if not args.output:
            print("Ошибка: для файлового режима необходимо указать выходной файл (-o)")
            return
        
        process_tsv_file(args.input, args.output, args.column)
        return
    
    # Если аргументов нет, показываем помощь
    parser.print_help()


if __name__ == "__main__":
    main()
