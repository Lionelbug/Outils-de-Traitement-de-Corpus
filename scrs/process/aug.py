import random
import jieba

# dictionnaire de synonyme
synonym_dict = {
    '不错': ['很好', '挺好', '可以'],
    '垃圾': ['不好', '很差', '糟糕'],
    '喜欢': ['爱', '热爱'],
    '推荐': ['引荐', '推介'],
    '画面': ['视觉', '图像']
}

def synonym_replacement(words, prob=0.3):
    new_words = []
    for word in words:
        if word in synonym_dict and random.random() < prob:
            new_word = random.choice(synonym_dict[word])
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def random_deletion(words, prob=0.1):
    if len(words) == 1:
        return words
    return [word for word in words if random.random() > prob]

def random_swap(words, num=1):
    new_words = words.copy()
    for _ in range(num):
        if len(new_words) < 2:
            break
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def augment_sentence(sentence: str) -> str:
    words = list(jieba.cut(sentence))
    if len(words) == 0:
        return sentence

    aug_type = random.choice(['sr', 'rd', 'rs'])  # 三选一

    if aug_type == 'sr':
        aug_words = synonym_replacement(words)
    elif aug_type == 'rd':
        aug_words = random_deletion(words)
    elif aug_type == 'rs':
        aug_words = random_swap(words)
    else:
        aug_words = words

    return ''.join(aug_words)

def augment_df(df: pd.DataFrame, num_aug: int = 1) -> pd.DataFrame:
    augmented_texts = []
    augmented_labels = []

    for _, row in df.iterrows():
        text = row['text']
        label = row['labels']

        # original sample
        augmented_texts.append(text)
        augmented_labels.append(label)

        # augmented sample
        for _ in range(num_aug):
            aug_text = augment_sentence(text)
            augmented_texts.append(aug_text)
            augmented_labels.append(label)

    return pd.DataFrame({'text': augmented_texts, 'labels': augmented_labels})