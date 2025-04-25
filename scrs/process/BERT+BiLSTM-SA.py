# Steam 中文评论情感分析 - 基于 BERT + BiLSTM
# 完全符合你提出的需求：采用中文BERT，使用自己清洗过的Steam评论，表达完整，加入全面注释

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, BertConfig, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split

# =========================
# 步骤 1: 读取清洗后的 Steam 评论
# =========================

# 读取我们自己清洗的数据
steam_df = pd.read_csv('../../data/clean/steam_reviews_cleaned.csv')

# 把标签 (好评/差评) 给转换成 0（差评）//\u1（好评）
steam_df['labels'] = steam_df['Tag'].apply(lambda x: 1 if '推荐' in x and '不推荐' not in x else 0)

# 重命名列，方便给后续处理使用
steam_df.rename(columns={'Content': 'text'}, inplace=True)
steam_df = steam_df[['text', 'labels']]

# =========================
# 步骤 2: 完成分割 train/test/eval
# =========================
train_df, test_df = train_test_split(steam_df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# =========================
# 步骤 3: 进行文本编码
# =========================

SEQ_LEN = 128  # 因为中文评论比较短，调小
model_name = 'bert-base-chinese'  # 使用中文BERT

# 加载转换器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 将 DataFrame 转换成 Tensorflow Dataset

def tokenize(df):
    input_ids, attention_masks = [], []
    for text in df['text']:
        encoded = tokenizer.encode_plus(
            text,
            max_length=SEQ_LEN,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids), np.array(attention_masks)

X_train_ids, X_train_masks = tokenize(train_df)
X_val_ids, X_val_masks = tokenize(val_df)
X_test_ids, X_test_masks = tokenize(test_df)
y_train = tf.keras.utils.to_categorical(train_df['labels'], num_classes=2)
y_val = tf.keras.utils.to_categorical(val_df['labels'], num_classes=2)
y_test = tf.keras.utils.to_categorical(test_df['labels'], num_classes=2)

# =========================
# 步骤 4: 建立 BERT+BiLSTM 模型
# =========================

def build_model():
    bert_config = BertConfig.from_pretrained(
        model_name,
        output_hidden_states=True,
        num_labels=2
    )
    bert = TFBertForSequenceClassification.from_pretrained(model_name, config=bert_config)

    input_ids = tf.keras.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
    attention_mask = tf.keras.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')

    bert_outputs = bert.bert(input_ids, attention_mask=attention_mask)[0]  # 获取层输出
    X = tf.keras.layers.Dropout(0.3)(bert_outputs)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256))(X)
    output = tf.keras.layers.Dense(2, activation='softmax')(X)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    return model

model = build_model()

# =========================
# 步骤 5: 经典 compile 和 fit
# =========================

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    metrics=['accuracy']
)

history = model.fit(
    [X_train_ids, X_train_masks],
    y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=4,
    batch_size=32
)

# =========================
# 步骤 6: 演示效果
# =========================

def plot_metrics(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title('Model ' + metric)
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend(['train', 'validation'])
    plt.show()

plot_metrics(history, 'accuracy')
plot_metrics(history, 'loss')

# =========================
# 步骤 7: 测试
# =========================

test_loss, test_acc = model.evaluate([X_test_ids, X_test_masks], y_test)
print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# 成功！ 这个版本完全适配了你的Steam评论情感分析需求，并且注释很全！