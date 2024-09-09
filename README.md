# AWS DeepRacer stanCode 201 Mar2024 B組 July 參賽紀錄

這個repo紀錄stanCode201 Mar2024 AI班 B組參與DeepRacer 7月份的過程，我會詳細記錄如何一步一步執行DeepRacer Community提供的deepracer-for-cloud(DRfC)的訓練，以及說明我們小組的策略。

README會包含以下內容：
1. 參賽結果
2. AWS DeepRacer介紹
3. 如何進行DRfC的訓練
4. 總花費
5. Reward function
6. Action Space
7. Hyperparameters
8. Log Anlysis

## 1.參賽結果
![stanCode小組七月分參賽結果](upload_img/DeepRacer_Jul_Result.jpeg)
感謝同組夥伴Blair, 孟勳, 宇韜, 沛融還有mentor南哥，最重要的是全額贊助我們的stanCode創辦人Jerry，讓我們可以不用擔心AWS貴鬆鬆的花費盡情的訓練！
我們組別在七月份的競賽結果為全台第四名，以及全球前一百名！！！

## 2.AWS DeepRacer介紹
關於DeepRacer是什麼，以及競賽規則說明可以參考[官方網站的說明](https://docs.aws.amazon.com/zh_tw/deepracer/latest/developerguide/what-is-deepracer.html)。
在此會著重在DeepRacer Community中常常提到的專有名詞以及其應用的RL原理進行說明：
![RL與監督式學習的比較](upload_img/SL_RL.png)

SL跟RL這兩種訓練方式的訓練資料很不一樣，監督式學習的資料有固定的答案，所以我們的目標就是盡量減少判斷的錯誤，因此我們會用Gradient Decent的方式讓誤差隨著訓練而下降。
但RL的學習資料則是有隨機性，例如在DeepRacer競賽中，車子現在行進到某個位置，你此時的state，也就是在彎中的角度、距離賽道的邊界也都會不一樣。
因此，在上面的RL示意圖中，agent，也就是車子，必須透過跟賽道environment的互動，利用action來確認現在是不是恰當的動作，藉此得到reward。
那什麼是恰當的動作？這個部分我們可以透過設定reward function來引導，例如我們會懲罰超出賽道、獎勵彎道中保持方向盤角度，或是保持越快的速度有越高的reward。
而車子每完成一次action後就會改變環境，作為新的input提供給agent。
透過這樣的訓練過程，我們會得到車子的policy，也可以想像成是開車的風格，風格可以決定我們面對當下環境要做什麼action，看是要激進過彎還是保守過彎。
最終我們透過計算Reward的總和來評價模型表現的好壞。
但要用什麼演算法達到RL的最佳模型？在DeepRacer中，絕大多數的人都會選擇PPO（Proximal Policy Optimization）演算法，他可以透過限制每次更新的幅度，來保持訓練過程的穩定性和效率。

AWS在這個競賽中提供兩種常見的演算法：PPO 和 SAC，主要有三個不同點
* Action Space兼容性：PPO 適用於離散和連續action space，而 SAC 主要適用於連續型。
* 學習方法：PPO 是 on-policy 算法，即從當前策略的觀察中學習，也因此實時數據需求非常大；SAC 是 off-policy 算法，可以利用先前策略的觀察進行學習。
* Entropy的處理：PPO 使用Entropy regularization來鼓勵探索，防止過早收斂，喪失探索性；SAC 通過在最大化目標中添加Entropy來平衡探索與過早收斂。

我們在DeepRacer的社群中發現大多數人都會選擇PPO，也是我們主要選擇的算法，我們歸納出以下幾個原因：
* PPO在policy上可以避免用過時或不相關的數據，因此可以保持連貫性跟穩定性。
* 數據跟策略的更新是同步的，所以可以減少數據分佈變化引起的不確定性。
* 第三，PPO會限制每次policy的更新都是小幅度的，因此可以確保模型的更新較為平穩。
* 最後PPO會根據當前最準確且最相關的數據來進行更新，可以確保學習的效率。

## 3.如何進行DRfC的訓練
## 4.總花費
## 5.Reward function
## 6.Action Space
## 7.Hyperparameters
## Log Anlysis
