---
layout: post
title:  " [코드리뷰] TFFM: TensorFlow FactorizationMachine"
date:   2019-10-18 21:19:41 +0900
categories: Code Review
comments: true
share: true
featured: true
---

### 데이터 셋 설명
>RecSys Challenge 2015 데이터
클릭 사건의 연속적 집합을 보여준다; 클릭 세션<br>
몇몇 클릭 세션은 구매로 이어지기도 한다. <br>
목적은 해당 유저가 구매를 할지 안할지 예측하는 것. 만약 구매를 한다면 어떤 상품을 구매할지?<br>
유저에게 더욱 정교한 프로모션, 할인 등을 제공한다.

>A detailed description of the challenge can be found on the website of the RecSys Challenge 2015.
Accepted contributions will be presented during the RecSys Challenge 2015 Workshop.

### 데이터 전처리 과정 리뷰
- yoochoose-buy 데이터에는 유저가 구매한 해당 아이템의 Item ID, 해당 아이템의 Category, 구매시기 Timestamp, 구매한 양 **Quantity 정보가 포함 돼있음<br>**

- yoochoose-click 데이터에는 구매 정보는 없으며, Timestamp, Item ID, Category 정보만 포함하고 있음. **Quantity 없음**

---
**Initial Data:** <br>

Session ID 는 index로 지정하자
```python
initial_buys_df.set_index('Session ID', inplace=True)
initial_clicks_df.set_index('Session ID', inplace=True)
```

Timestamp 정보는 이번 예시에서 사용하지 않는다.
~~~python
initial_buys_df = initial_buys_df.drop('Timestamp', 1)
initial_clicks_df = initial_clicks_df.drop('Timestamp', 1)
~~~

상위 10000명의 구매 유저만 사용하자.
Collector 모듈의 Counter 함수를 사용해서, initial_buys_df 데이터 프레임의 Session ID 의 빈도를 확인하고, 가장 높은 10000 개의 데이터를 추출한다. 이후에 해당 10000개의 카운트 데이터를 딕셔너리 데이터 타입으로 변경한 이후에, 해당 Key 값들을 저장함으로써 가장 구매 수가 많은 유저들의 번호를 얻는 방식이다.

~~~python
x = Counter(initial_buys_df.index).most_common(10000)
top_k = dict(x).keys()
initial_buys_df = initial_buys_df[initial_buys_df.index.isin(top_k)]
initial_clicks_df = initial_clicks_df[initial_clicks_df.index.isin(top_k)]
~~~

initial_buys_df 의 Session ID는 one-hot 인코딩을 적용할 것이므로, 해당 컬럼은 따로 복사를 해두자

~~~python
initial_buys_df['_Session ID'] = initial_buys_df.index
~~~

---
**One-Hot Encoding (Dummies) :**<br>

어느 정도 정리가 된 1차 데이터 프레임 (initial_buys_df 와 initial_clicks_df)에 대해 모두 더미화 시키는 게 필요하다.
~~~python
transformed_buys = pd.get_dummies(initial_buys_df)
transformed_clicks = pd.get_dummies(initial_clicks_df)
~~~~

더미화된 데이터프레임 transformed_buys와 transformed_clicks은 item ID와 Quantity를 제외한 Category 컬럼에 대해 one-hot encoding이 진행됐다. item ID와 quantity는 현재 카테고리 데이터가 아니므로 인코딩에서 제외된다. 이후에, filtered 데이터 프레임들은 'Item' 또는 'Category'를 포함하는 컬럼들로 filtering 한 것이다.

~~~python
filtered_buys = transformed_buys.filter(regex="Item.*|Category.*")
filtered_clicks = transformed_clicks.filter(regex="Item.*|Category.*")

historical_buy_data = filtered_buys.groupby(filtered_buys.index).sum()
historical_buy_data = historical_buy_data.rename(columns=lambda column_name: 'buy history:' + column_name)

historical_click_data = filtered_clicks.groupby(filtered_clicks.index).sum()
historical_click_data = historical_click_data.rename(columns=lambda column_name: 'click history:' + column_name)
~~~

위의 historical data들은 buy.index와 clicks.index로 그루핑 이후 해당 window에서 값들을 모두 더해주기 때문에, 각각 10000개씩의 관측 값들을 지니게 된다. 또한 각각의 데이터가 구매 데이터(...buy_data)에서 유래한 데이터인지, 클릭 데이터(...click_data)에서 유래한 데이터인지를 레이블링 하기 위해 각각의 컬렴 명에 buy history 또는 click history 이름을 추가해주었다. 해당 레이블링을 활용해서 추후에 full information 데이터셋과 cold-start 데이터 셋으로 구분할 수 있게 된다.

---
**Merging Data :**<br>

해당 merge step을 통해 어떤 데이터 프레임을 얻고 싶었던 걸까? 판다스의 merge는 기본적으로 서로 다른 테이블의 같은 컬럼명을 찾고 해당 컬럼들에 대해 inner join 을 시행하게 돼있으므로, index가 같은 Session들에 한해 단순히 column append를 진행하게 된다.

~~~python
merged1 = pd.merge(transformed_buys, historical_buy_data, left_index=True, right_index=True)
merged

merged2.drop(['Item ID', '_Session ID', 'click history:Item ID', 'buy history:Item ID'], 1, inplace=True)
~~~

---

**Train, Test Set Splitting :**<br>

1차적으로 트레이닝 셋과 테스트 셋의 X, y 값들로 split 이후, 2차적으로 테스트 값들은 full information 케이스와 cold-start 케이스로 나누게 된다. cold-start 케이스의 경우, 카테고리 데이터만을 사용하고 그렇지 않은 컬럼들은 배제시키는 방향으로 for 문 안에서 해당 값들은 0으로 처리해주게 된다.

~~~python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

X_te, X_te_cs, y_te, y_te_cs = train_test_split(X_te, y_te, test_size=0.5)
X_te_cs = pd.DataFrame(X_te_cs, columns=merged2.columns)

for column in X_te_cs.columns:
    if ('buy' in column or 'click' in column) and ('Category' not in column):
        X_te_cs[column] = 0
~~~

### 모델링

~~~python
model = TFFMRegressor(
    order=2,
    rank=7,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
    n_epochs=100,
    batch_size=-1,
    init_std=0.001,
    input_type='dense'
)


model.fit(X_tr, y_tr, show_progress=True)
predictions = model.predict(X_te)
cold_start_predictions = model.predict(X_te_cs)
print('MSE: {}'.format(mean_squared_error(y_te, predictions)))
print('Cold-start MSE: {}'.format(mean_squared_error(y_te_cs, predictions)))
model.destroy()
~~~
