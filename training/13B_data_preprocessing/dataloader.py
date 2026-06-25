"""
AI-Hub 한영 병렬 코퍼스 통합 DataLoader
- 4가지 데이터셋 지원
- Tower 모델 학습 형식으로 변환
"""

import json
import os
from pathlib import Path
from typing import Iterator, Dict, List, Optional
from dataclasses import dataclass
import random


@dataclass
class TranslationPair:
    """번역 쌍 데이터 클래스"""
    source: str  # 영어
    target: str  # 한국어
    domain: str
    direction: str  # "en2ko" or "ko2en"


class DatasetType:
    """데이터셋 타입 상수"""
    CONFERENCE = "conference"  # 국제 학술회의용
    TECH_PARALLEL = "tech_parallel"  # 기술과학 분야 병렬
    TECH_SCIENCE = "tech_science"  # 한국어-영어 (기술과학)
    SOCIAL_SCIENCE = "social_science"  # 한국어-영어 (사회과학)


class ConferenceDataLoader:
    """
    데이터셋 1: 국제 학술회의용 전문분야 한영-영한 통번역 데이터
    JSON 구조: {"data": [{"source_cleaned": "...", "MTPE": "...", ...}]}
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load(self) -> Iterator[TranslationPair]:
        """JSON 파일들을 순회하며 번역 쌍 반환"""
        from tqdm import tqdm
        json_files = list(self.data_dir.rglob("*.json"))

        for json_file in tqdm(json_files, desc="  Processing files"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                items = data.get('data', data) if isinstance(data, dict) else data
                
                for item in items:
                    source = item.get('source_cleaned', '').strip()
                    target = item.get('MTPE', '').strip()
                    domain = item.get('subdomain', item.get('domain', 'academic'))
                    
                    if source and target:
                        yield TranslationPair(
                            source=source,
                            target=target,
                            domain=domain,
                            direction="en2ko"
                        )
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue


class TechParallelDataLoader:
    """
    데이터셋 2: 기술과학 분야 한-영 번역 병렬 말뭉치 데이터
    JSON 구조: {"data": [{"en": "...", "ko": "...", "source_language": "...", ...}]}
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> Iterator[TranslationPair]:
        from tqdm import tqdm
        json_files = list(self.data_dir.rglob("*.json"))

        for json_file in tqdm(json_files, desc="  Processing files"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                items = data.get('data', data) if isinstance(data, dict) else data
                
                for item in items:
                    en = item.get('en', '').strip()
                    ko = item.get('ko', '').strip()
                    source_lang = item.get('source_language', 'en')
                    domain = item.get('subdomain', item.get('domain', 'tech'))
                    
                    if en and ko:
                        # source_language에 따라 방향 결정
                        if source_lang == 'en':
                            yield TranslationPair(
                                source=en,
                                target=ko,
                                domain=domain,
                                direction="en2ko"
                            )
                        else:
                            yield TranslationPair(
                                source=en,
                                target=ko,
                                domain=domain,
                                direction="ko2en"
                            )
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue


class TechScienceDataLoader:
    """
    데이터셋 3: 한국어-영어 번역 말뭉치 (기술과학)
    JSON 구조: {"data": [{"en": "...", "ko": "...", ...}]}
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> Iterator[TranslationPair]:
        from tqdm import tqdm
        json_files = list(self.data_dir.rglob("*.json"))

        for json_file in tqdm(json_files, desc="  Processing files"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                items = data.get('data', data) if isinstance(data, dict) else data
                
                for item in items:
                    en = item.get('en', '').strip()
                    ko = item.get('ko', '').strip()
                    source_lang = item.get('source_language', 'ko')
                    domain = item.get('subdomain', item.get('domain', 'tech_science'))
                    
                    if en and ko:
                        direction = "en2ko" if source_lang == 'en' else "ko2en"
                        yield TranslationPair(
                            source=en,
                            target=ko,
                            domain=domain,
                            direction=direction
                        )
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue


class SocialScienceDataLoader:
    """
    데이터셋 4: 한국어-영어 번역 말뭉치 (사회과학)
    JSON 구조: {"data": [{"en": "...", "ko": "...", ...}]}
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> Iterator[TranslationPair]:
        from tqdm import tqdm
        json_files = list(self.data_dir.rglob("*.json"))

        for json_file in tqdm(json_files, desc="  Processing files"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                items = data.get('data', data) if isinstance(data, dict) else data
                
                for item in items:
                    en = item.get('en', '').strip()
                    ko = item.get('ko', '').strip()
                    source_lang = item.get('source_language', 'ko')
                    domain = item.get('subdomain', item.get('domain', 'social_science'))
                    
                    if en and ko:
                        direction = "en2ko" if source_lang == 'en' else "ko2en"
                        yield TranslationPair(
                            source=en,
                            target=ko,
                            domain=domain,
                            direction=direction
                        )
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue


class UnifiedDataLoader:
    """
    모든 데이터셋을 통합하여 로드하는 클래스
    """
    
    def __init__(self, data_config: Dict[str, str]):
        """
        Args:
            data_config: {데이터셋_타입: 경로} 딕셔너리
        """
        self.loaders = {}
        
        if DatasetType.CONFERENCE in data_config:
            self.loaders[DatasetType.CONFERENCE] = ConferenceDataLoader(
                data_config[DatasetType.CONFERENCE]
            )
        
        if DatasetType.TECH_PARALLEL in data_config:
            self.loaders[DatasetType.TECH_PARALLEL] = TechParallelDataLoader(
                data_config[DatasetType.TECH_PARALLEL]
            )
        
        if DatasetType.TECH_SCIENCE in data_config:
            self.loaders[DatasetType.TECH_SCIENCE] = TechScienceDataLoader(
                data_config[DatasetType.TECH_SCIENCE]
            )
        
        if DatasetType.SOCIAL_SCIENCE in data_config:
            self.loaders[DatasetType.SOCIAL_SCIENCE] = SocialScienceDataLoader(
                data_config[DatasetType.SOCIAL_SCIENCE]
            )
    
    def load_all(self, direction_filter: Optional[str] = None) -> Iterator[TranslationPair]:
        """
        모든 데이터셋에서 번역 쌍 로드 (Generator 방식으로 메모리 절약)

        Args:
            direction_filter: "en2ko" 또는 "ko2en" (None이면 모두 포함)

        Yields:
            TranslationPair 객체
        """
        for dataset_type, loader in self.loaders.items():
            print(f"Loading {dataset_type}...")
            count = 0
            for pair in loader.load():
                if direction_filter is None or pair.direction == direction_filter:
                    yield pair
                    count += 1
            print(f"  Loaded {count:,} pairs from {dataset_type}")
    
    def load_as_huggingface_dataset(
        self,
        direction: str = "en2ko",
        train_ratio: float = 0.95,
        seed: int = 42,
        batch_size: int = 50000
    ):
        """
        Hugging Face Dataset 형식으로 변환 (메모리 효율적 방식)

        Args:
            direction: "en2ko" (영→한) 또는 "ko2en" (한→영)
            train_ratio: 학습 데이터 비율
            seed: 랜덤 시드
            batch_size: 배치 크기 (메모리 관리)

        Returns:
            DatasetDict with 'train' and 'validation' splits
        """
        from datasets import Dataset, DatasetDict
        import pyarrow as pa
        import pyarrow.parquet as pq
        import tempfile
        import os

        print(f"\nConverting to HuggingFace dataset format...")

        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        temp_parquet = os.path.join(temp_dir, "data.parquet")

        # Arrow 테이블 스키마 정의
        schema = pa.schema([
            ('source', pa.string()),
            ('target', pa.string()),
            ('source_lang', pa.string()),
            ('target_lang', pa.string()),
            ('domain', pa.string()),
        ])

        # Parquet 파일에 배치 단위로 쓰기
        writer = None
        batch = []
        total_count = 0

        try:
            for pair in self.load_all(direction_filter=direction):
                if direction == "en2ko":
                    item = {
                        "source": pair.source,
                        "target": pair.target,
                        "source_lang": "English",
                        "target_lang": "Korean",
                        "domain": pair.domain
                    }
                else:  # ko2en
                    item = {
                        "source": pair.target,
                        "target": pair.source,
                        "source_lang": "Korean",
                        "target_lang": "English",
                        "domain": pair.domain
                    }

                batch.append(item)
                total_count += 1

                # 배치가 가득 차면 Parquet에 쓰기
                if len(batch) >= batch_size:
                    table = pa.Table.from_pylist(batch, schema=schema)
                    if writer is None:
                        writer = pq.ParquetWriter(temp_parquet, schema)
                    writer.write_table(table)
                    batch = []
                    print(f"  Processed {total_count:,} pairs...")

            # 남은 데이터 쓰기
            if batch:
                table = pa.Table.from_pylist(batch, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(temp_parquet, schema)
                writer.write_table(table)

            if writer:
                writer.close()

            print(f"Total: {total_count:,} pairs")

            # Parquet 파일에서 Dataset 로드 (메모리 효율적)
            print("Loading dataset from Parquet...")
            dataset = Dataset.from_parquet(temp_parquet)

            # 셔플
            print("Shuffling dataset...")
            dataset = dataset.shuffle(seed=seed)

            # Train/Validation 분할
            print("Splitting into train/validation...")
            split_dict = dataset.train_test_split(train_size=train_ratio, seed=seed)

            result = DatasetDict({
                'train': split_dict['train'],
                'validation': split_dict['test']
            })

            print(f"  Train: {len(result['train']):,} examples")
            print(f"  Validation: {len(result['validation']):,} examples")

            return result

        finally:
            # 임시 파일 삭제
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


def format_for_tower(example: Dict, tokenizer) -> Dict:
    """
    Tower 모델 학습을 위한 프롬프트 형식 변환
    
    Tower 형식:
    <|im_start|>user
    Translate the following text from {source_lang} into {target_lang}.
    {source_lang}: {source_text}
    {target_lang}:<|im_end|>
    <|im_start|>assistant
    {target_text}<|im_end|>
    """
    source_lang = example['source_lang']
    target_lang = example['target_lang']
    source_text = example['source']
    target_text = example['target']
    
    # 프롬프트 구성
    prompt = f"""<|im_start|>user
Translate the following text from {source_lang} into {target_lang}.
{source_lang}: {source_text}
{target_lang}:<|im_end|>
<|im_start|>assistant
{target_text}<|im_end|>"""
    
    return {"text": prompt}


def create_dataset_for_training(
    data_config: Dict[str, str],
    direction: str = "en2ko",
    output_dir: str = "./processed_data",
    train_ratio: float = 0.95,
    seed: int = 42
):
    """
    학습용 데이터셋 생성 및 저장
    
    Args:
        data_config: 데이터셋 경로 설정
        direction: 번역 방향
        output_dir: 출력 디렉토리
        train_ratio: 학습 데이터 비율
        seed: 랜덤 시드
    """
    loader = UnifiedDataLoader(data_config)
    dataset = loader.load_as_huggingface_dataset(
        direction=direction,
        train_ratio=train_ratio,
        seed=seed
    )
    
    # 저장
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    
    print(f"\nDataset saved to {output_dir}")
    print(f"  Train: {len(dataset['train']):,} examples")
    print(f"  Validation: {len(dataset['validation']):,} examples")
    
    return dataset


# 사용 예시
if __name__ == "__main__":
    # 데이터 경로 설정 (사용자 환경에 맞게 수정)
    DATA_CONFIG = {
        DatasetType.CONFERENCE: "/path/to/005.국제 학술회의용 전문분야 한영-영한 통번역 데이터",
        DatasetType.TECH_PARALLEL: "/path/to/026.기술과학 분야 한-영 번역 병렬 말뭉치 데이터",
        DatasetType.TECH_SCIENCE: "/path/to/한국어-영어 번역 말뭉치 (기술과학)",
        DatasetType.SOCIAL_SCIENCE: "/path/to/한국어-영어 번역 말뭉치 (사회과학)",
    }
    
    # 데이터셋 생성
    dataset = create_dataset_for_training(
        data_config=DATA_CONFIG,
        direction="en2ko",  # 영어 → 한국어
        output_dir="./processed_data_en2ko",
        train_ratio=0.95
    )
    
    # 샘플 확인
    print("\n=== Sample ===")
    print(dataset['train'][0])
