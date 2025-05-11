AI 比賽全範圍主題深入學習指南

人工智慧競賽全方位學習指南
準備人工智慧（AI）競賽需要全面掌握理論知識與實作技能。本指南將依據常見主題結構，深入說明每個子主題的理論概念、實作方法、Python 範例（使用 NumPy、Pandas、Scikit-learn、PyTorch 等），並提供小技巧與常見錯誤提醒。所有內容均以繁體中文呈現，方便您系統化學習。
編程基礎 (Programming Basics)
AI 實作離不開程式設計，本節涵蓋 Python 語言基礎及常用資料科學庫的使用，包括 NumPy 的數值計算和 Pandas 的資料處理技巧。
Python 基礎語法與資料結構
理論概念： Python 是資料科學領域的主流語言。熟悉其基本語法（變數、資料型態、條件控制、迴圈等）以及內建資料結構（如清單List、元組Tuple、字典Dict和集合Set）相當重要。瞭解這些資料結構的特性與操作方法有助於撰寫高效的代碼。例如，List 可儲存有序序列且可變動，Dict 可用於鍵值對快速查找。 實作方法： 建議使用互動式環境（如 Jupyter Notebook）進行練習。熟悉 Python 標準函式庫及常用語法如列表生成式(List Comprehension)等。以下是一個簡單的 Python 基本操作範例，包括變數指定和列表操作：
python
複製
編輯
# Python 基本範例：計算奇數平方和
numbers = list(range(1, 11))            # 建立 1 到 10 的清單
odd_squares = [x**2 for x in numbers if x % 2 == 1]  # 列表生成式取奇數的平方
print("奇數平方列表:", odd_squares)
print("奇數平方和:", sum(odd_squares))
上述程式會輸出奇數的平方列表以及這些平方數的總和。透過簡潔的列表生成式，我們快速計算了 1 到 10 的奇數平方和。 小技巧與常見錯誤：
代碼可讀性： 盡量使用有意義的變數名稱並保持縮排一致，增進程式可讀性。
資料結構選擇： 根據需求選擇合適的資料結構。例如，需要順序且頻繁新增刪除用 list，需要快速查找用 dict或set。
常見錯誤： 注意 Python 的縮排規則，錯誤的縮排會導致 IndentationError。另外，留意變數的型別，例如將字串轉為整數再進行數學運算，否則可能出現型別錯誤。
NumPy 與向量化計算
理論概念： NumPy 提供高效的多維陣列物件 (ndarray) 及向量化運算。在數值計算中，比起 Python 原生的 list，使用 NumPy 陣列能大幅提升運算效率
zh.wikipedia.org
。NumPy 支援元素層級的運算 (element-wise operations)，這意味著我們可以對整個陣列直接進行數學計算而不需要使用 Python loop。 實作方法： 使用 numpy 模組來建立陣列、進行線性代數計算等。以下範例展示如何利用 NumPy 進行向量化運算，相較傳統 Python 迴圈的效率：
python
複製
編輯
import numpy as np

# 建立一個包含100萬個隨機數的陣列
arr = np.random.rand(1000000)
# 利用向量化操作計算每個元素的平方
squares = arr ** 2

print("陣列前5個元素:", arr[:5])
print("平方後前5個元素:", squares[:5])
這段程式使用 NumPy 產生100萬個隨機數，並以向量化方式計算平方。在 NumPy 中，arr ** 2 會對陣列中的每個元素計算平方，執行底層的高效C運算，相較以 Python 迴圈逐個計算要快得多。 小技巧與常見錯誤：
向量化優勢： 儘量用陣列的向量化操作取代純 Python 的 loop，可以取得顯著效能提升。
矩陣維度： NumPy 操作常涉及矩陣維度，注意對齊 (broadcasting) 規則，確保運算時陣列維度相容，否則會引發錯誤。

常見錯誤： Python 的 list 和 NumPy ndarray 是不同型別，直接對 Python list 使用 NumPy 函式可能導致錯誤。若遇到類型不匹配問題，可用 np.array(list) 將 list 轉為 ndarray。
Pandas 與資料處理
理論概念： Pandas 提供 DataFrame 資料結構，使我們能方便地操作結構化表格資料。可將其視為 Excel 試算表或SQL資料表的程式介面。Pandas 支援資料篩選、清理、聚合等操作，是資料前處理與分析的利器。 實作方法： 常見操作包括讀取資料（例如 CSV 檔）、篩選欄位、處理缺失值和描述性統計等。下面以 Iris 鳶尾花資料集為例，展示如何使用 Pandas 進行資料載入與基本分析：
python
複製
編輯
import pandas as pd
from sklearn.datasets import load_iris

# 載入 Iris 鳶尾花資料集並轉為 DataFrame
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target  # 新增物種欄位 (0,1,2 對應三種鳶尾花)
# 簡單將 species 數值轉成人類可讀的名稱
species_mapping = {i: name for i, name in enumerate(iris.target_names)}
df['species'] = df['species'].map(species_mapping)

print("資料集前5筆：")
print(df.head())          # 查看前5筆資料
print("\n各品種樣本數：")
print(df['species'].value_counts())  # 計算每個物種的樣本數
在上述範例中，我們使用 sklearn.datasets 提供的 Iris 資料，透過 Pandas 載入為 DataFrame 並命名欄位。接著印出前5筆資料查看內容，並統計每個品種的樣本數。這些操作展示了 Pandas 在資料探索 (EDA) 階段的便利性。 小技巧與常見錯誤：
索引與選擇： Pandas 提供 loc（標籤索引）與 iloc（整數位置索引）兩種取值方式。混用索引容易導致錯誤，要清楚何時用哪一種。
處理缺失值： 在資料清理時，可使用 df.dropna() 移除含缺失值的列，或用 df.fillna() 以統計值填補缺失。忘記處理缺失值可能導致之後的模型訓練出錯。
常見錯誤： DataFrame 切片操作時，df[a:b]會取行範圍，但 df['col'] 取列。容易誤將小括號寫成中括號導致 KeyError。務必注意索引符號的使用。
資料前處理與特徵工程 (Data Preprocessing & Feature Engineering)
在將資料送入機器學習模型前，必須對資料進行適當的前處理和特徵工程。良好的資料處理可以提高模型的表現。本節介紹資料清理、特徵縮放、編碼與選擇等技巧。
資料清理與缺失值處理
理論概念： 現實世界的資料常包含缺失值、異常值或不一致的格式。必須先清理資料，例如處理缺失值、移除異常點、修正錯誤格式等，以確保後續分析的可靠性。缺失值處理方法主要有刪除或填補：刪除可能導致資訊流失，而填補則需謹慎選擇策略（平均值、中位數、眾數或插值等）。 實作方法： 使用 Pandas 可以輕鬆偵測和處理缺失值。以下舉例說明如何檢查缺失與進行填補：
python
複製
編輯
# 建立範例 DataFrame，包含缺失值 NaN
data = {'年齡': [25, 30, None, 40], '薪資': [50000, None, 45000, 52000]}
df = pd.DataFrame(data)
print("原始資料：\n", df)

# 檢查缺失值
print("\n缺失值計數：")
print(df.isna().sum())  # 計算每欄位缺失值數量

# 用平均值填補缺失值
df_filled = df.fillna(df.mean(numeric_only=True))
print("\n填補缺失值後：\n", df_filled)
上述程式先建立一個含缺失值的 DataFrame。df.isna().sum() 用於檢視各欄缺失值數量。接著用 df.fillna() 將缺失的數值欄位以平均值填補（numeric_only=True 參數確保只對數值欄位取平均）。輸出結果顯示缺失值已被替換為該欄的平均值。 小技巧與常見錯誤：
缺失值指標： 有些資料集缺失值用特殊標記（如 "NA", "NULL" 或特殊數字），需先轉換為 NaN 再統一處理。
填補影響： 用統計值填補可能降低資料變異，需考量對模型的影響。另一種策略是增添缺失值指示特徵（flag），標記該樣本此欄原本缺值。
常見錯誤： Pandas 的 mean()、median() 等預設會忽略 NaN 直接計算。但如果資料中缺失比例很高，盲目填補不一定合理，甚至可能引入偏差。
特徵縮放與正規化
理論概念： 特徵縮放（Feature Scaling）旨在將不同量綱的特徵轉換到相近的範圍，以避免某些數值大的特徵主導模型訓練
ntudac.medium.com
。常用的縮放方法有標準化 (Standardization) 和常態化 (Normalization)
ntudac.medium.com
：
標準化： 將特徵數值轉換為平均數0、標準差1的分佈，即 $z$-score 標準化： $z = (x - \mu) / \sigma$。
常態化： 又稱最小最大縮放，將數值線性轉換到 [0,1] 範圍： $x_{norm} = (x - x_{min})/(x_{max}-x_{min})$
ntudac.medium.com
。
縮放在使用梯度下降優化的模型中特別重要，否則特徵尺度差異會導致收斂變慢或陷入局部最優。 實作方法： 使用 Scikit-learn 的 StandardScaler 或 MinMaxScaler 等工具可方便地對特徵進行縮放。範例：
python
複製
編輯
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 模擬原始特徵資料：例如年齡和收入
X = np.array([[25, 50000], [30, 80000], [45, 120000]], dtype=float)
print("原始特徵：\n", X)

# Min-Max 常態化
mms = MinMaxScaler()
X_norm = mms.fit_transform(X)
print("\nMin-Max 常態化結果：\n", X_norm)

# Z-score 標準化
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
print("\nZ-score 標準化結果：\n", X_std)
此程式建立一組「年齡、收入」特徵矩陣。Min-Max 正規化將兩個特徵縮放到 [0,1]，而 Z-score 標準化則轉換為均值0、標準差1的數值。輸出結果可觀察到不同縮放方法的效果：年齡和收入被帶到相近的尺度範圍，避免後續分析時數值差異過大。 小技巧與常見錯誤：
訓練與測試一致： **重要：**縮放參數應只從訓練資料計算，並應用於測試資料，以避免資料洩漏。使用 Sklearn 時，先 fit 于訓練集，再用 transform 處理測試集。
例外情況： 僅對需要的模型進行縮放。像決策樹和隨機森林對特徵尺度不敏感，不縮放也可；而使用距離度量的模型（如 KNN、SVM）和使用梯度的模型（如迴歸、神經網路）通常需要縮放。
常見錯誤： 忘記在部署時對新輸入進行相同的縮放轉換，導致模型預測不準確。應保存縮放器（Scaler）的參數，用於未來資料處理。
類別編碼與特徵工程
理論概念： 對於類別型（分類）特徵，需將文字標籤轉為模型可處理的數字。標籤編碼將類別以序號表示，但對無序類別直接給予數值會產生虛假的大小關係。獨熱編碼 (One-Hot Encoding) 則為每個類別建立二元指標變數，能避免順序誤導。在特徵工程方面，可能需要對原始特徵做變換或組合，以提取有用資訊，如多項式特徵、交互特徵等。 實作方法： Scikit-learn 提供 LabelEncoder 和 OneHotEncoder 進行類別編碼。Pandas 的 get_dummies 也是方便的一種方式。下面展示使用 Pandas 進行 one-hot 編碼：
python
複製
編輯
import pandas as pd

# 範例資料集：性別和城市為類別特徵
df = pd.DataFrame({
    '性別': ['男', '女', '女', '男'],
    '城市': ['臺北', '臺中', '臺北', '高雄']
})
print("原始資料：\n", df)

# 使用 pandas 獨熱編碼
df_onehot = pd.get_dummies(df, columns=['性別', '城市'])
print("\n獨熱編碼後：\n", df_onehot)
上述資料包含「性別」和「城市」兩個類別欄位。pd.get_dummies 自動將這些欄位轉為獨熱編碼形式。例如，性別欄位轉為 性別_男 和 性別_女 兩個二元欄，城市則轉為 城市_臺北、城市_臺中、城市_高雄。編碼結果中每筆資料在其對應類別的欄位取值為1，其餘為0。 小技巧與常見錯誤：
避免虛假順序： 切勿對無序分類直接使用標籤編碼輸入模型，否則模型可能誤將其當作有序數值。無序類別應使用獨熱編碼。
高卡編碼： 當分類可能值很多時，獨熱編碼會導致維度爆炸。可考慮目標編碼等技巧或降維處理。
常見錯誤： 在訓練和測試資料集分別做獨熱編碼可能導致欄位不對齊。因此最好先合併資料做編碼或確保兩者採用相同的欄位集合。例如訓練集中沒有出現的城市在測試集出現，需要事先在訓練的編碼器中處理，否則模型將無法處理新類別。
特徵選擇與降維
理論概念： 特徵選擇指從眾多特徵中挑選最有用的子集，以減少過度擬合風險並提高模型效率。降維技術則是將高維特徵投影到較低維空間，保留關鍵資訊。主成分分析 (PCA) 是常用的降維方法，它透過找出資料中變異最大的方向（主成分）來壓縮資料維度
medium.com
。使用 PCA 可將原有高度相關的特徵轉成少數獨立的綜合特徵，同時降低資料維度
medium.com
。 實作方法： Scikit-learn 的 SelectKBest、SelectFromModel 等可用於特徵選擇；sklearn.decomposition.PCA 可方便地執行主成分分析。以下示範如何對特徵進行 PCA 降維：
python
複製
編輯
from sklearn.decomposition import PCA

# 使用先前的 iris 數據框的特徵欄位進行 PCA 將4維縮減到2維
X = df_onehot.values.astype(float)  # 這裡僅作示範，用 one-hot 處理後的資料
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print("降維後資料形狀:", X_reduced.shape)
print("前兩個主成分解釋的總變異比例: {:.2f}%".format(pca.explained_variance_ratio_.sum()*100))
上述代碼將資料從原始特徵空間轉換到2個主成分空間。輸出顯示降維後的形狀，以及前兩個主成分所解釋的變異比例（可理解為資訊保留程度）。在應用中，我們通常選擇能解釋大部分變異的前幾個主成分作為新的特徵。 小技巧與常見錯誤：
適度降維： PCA 等降維方法在保持資料主要資訊的同時，無法保留解釋性——主成分通常是原特徵的複雜組合，難以直觀解讀。因此在需要模型可解釋性的任務上要謹慎使用。
*特徵選擇: * 可以透過模型的特徵重要度（如隨機森林的feature_importances_）或統計檢定（如卡方檢定）來篩選特徵。移除對結果幾乎沒有影響的特徵可以簡化模型。
常見錯誤： 應在資料標準化之後進行 PCA，否則原始尺度差異大的特徵會主導第一主成分的方向，使降維結果偏差。
監督式學習 (Supervised Learning)
監督式學習指模型從帶有標籤的資料中學習，能對新輸入預測相對應的輸出
ai4dt.wordpress.com
ai4dt.wordpress.com
。根據預測目標類型不同，監督式學習又分為分類（離散類別）與迴歸（連續數值）問題。本節將介紹幾種主要的監督式演算法，包括其理論原理、實作及範例。
監督式學習概念總覽
理論概念： 在監督式學習中，我們提供模型一組輸入特徵 $X$ 及對應的目標值 $y$（標籤），讓模型學習 $X \to y$ 的映射關係
medium.com
。常見任務如圖片分類（輸入圖片特徵，輸出類別標籤）或房價預測（輸入房屋特徵，輸出價格）。監督式學習的優點是由於有標準答案，模型優化明確且通常預測準確率較高；缺點是依賴大量標記資料，標註過程可能耗時費力
medium.com
。 模型訓練與評估： 我們通常將資料分為訓練集、驗證集（可選）與測試集
medium.com
。模型在訓練集上學習，在驗證集上調整參數，在最終的測試集上評估泛化能力。模型學習的核心是誤差優化：定義一個損失函數（例如均方誤差或交叉熵），透過演算法（如梯度下降）調整模型參數以降低損失，進而提高預測準確率。 實作方法： Scikit-learn 提供了統一的 API 來使用各種監督式模型。我們可以輕鬆地 fit 模型並 predict 輸出結果。下面各小節將逐一介紹常見演算法。 小技巧與常見錯誤：
過擬合與正則化： 模型在訓練集表現很好但在測試集表現差被稱為過度擬合
ntudac.medium.com
。可透過正則化手段（如加入懲罰項 L1/L2、剪枝決策樹、Dropout 層等）降低此風險。
資料分佈變化： 監督式模型假設訓練與測試資料分佈相似。若日後應用場景資料分佈變化很大，需要額外的遷移學習或領域適應技術。
常見錯誤： 混淆分類與迴歸的模型。例如，試圖用線性迴歸處理分類問題或相反，會導致不合理的結果。需根據任務性質選擇正確模型。
接下來，我們將介紹各種監督式學習算法的詳細內容。
線性迴歸 (Linear Regression)
理論講解： 線性迴歸是最基本的迴歸模型，用於預測連續數值。它假設自變數 $X$ 和因變數 $y$ 存在線性關係，以線性函數來近似輸出
zh.wikipedia.org
。簡單線性迴歸的形式為 $y = w x + b$；多元線性迴歸則是 $y = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$。模型透過最小化殘差平方和（Ordinary Least Squares）來估計參數，使預測值與真實值之間的均方誤差最小
zh.wikipedia.org
。 線性迴歸直觀易懂且結果具有可解釋性（參數權重可解釋每個特徵對預測的影響）。然而，其假設輸出與各特徵間為線性關係，若資料呈現非線性特性，模型表現會較差。另外模型對異常值較為敏感。 應用場合： 線性迴歸廣泛用於預測各種數值，如房價、銷售額、體重等。例如，透過房屋面積、地點等特徵預測房價。它也是很多複雜模型的基石概念。 實作步驟： 使用 Scikit-learn 的 LinearRegression 非常簡便。通常步驟為：初始化模型、使用訓練資料 fit、再對新資料 predict。以下在糖尿病數據集上示範線性迴歸：
python
複製
編輯
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error

# 載入範例資料集（糖尿病資料集），並拆分為輸入特徵 X 和目標 y
X, y = load_diabetes(return_X_y=True)
# 使用前 400 筆資料訓練，後 42 筆資料測試
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

model = LinearRegression()
model.fit(X_train, y_train)   # 模型訓練
predictions = model.predict(X_test)  # 模型預測

# 評估模型的均方誤差
mse = mean_squared_error(y_test, predictions)
print("測試集均方誤差 MSE:", mse)
print("模型參數:", model.coef_)
在此範例中，我們用了 Scikit-learn 內建的糖尿病資料集來訓練線性迴歸模型，並輸出測試集上的均方誤差（MSE）作為性能指標。最後列印的 model.coef_ 是各特徵對預測的權重參數。 結果解讀： MSE 表示預測值與實際值的平均平方偏差，值越小代表模型預測越準確。線性迴歸的係數則反映每個特徵對目標的影響方向和程度，例如正係數表示該特徵增加會使預測值上升。 小技巧與常見錯誤：
線性假設檢驗： 線性迴歸有幾項基本假設，如線性關係、殘差獨立同分佈、無多重共線性等。可透過殘差圖等方法檢驗。例如殘差若呈現系統性趨勢，表示線性假設可能不成立。
特徵縮放與偏置： 線性迴歸本身不要求特徵縮放，但若帶有正則化（如 Ridge, Lasso），則通常需要對特徵做標準化處理以讓懲罰對不同尺度特徵有一致影響。
常見錯誤： 避免在訓練時使用未來資訊。例如，在預測房價時不應使用未來經濟指標訓練模型，否則導致未來資訊洩漏，模型在實際應用中會表現不佳。
邏輯迴歸 (Logistic Regression)
理論講解： 邏輯迴歸實際上是用於分類的模型（儘管名稱帶有「迴歸」）
vocus.cc
。它常用於解決二元分類問題，例如判斷某封郵件是否為垃圾郵件。邏輯迴歸模型通過邏輯斯谛函數 (Sigmoid) 將線性組合輸入轉換為介於0和1之間的機率值
vocus.cc
。輸出機率高於0.5則預測為正類（可調整臨界值）。其決策邏輯為：
𝑝
=
𝜎
(
𝑤
⋅
𝑥
+
𝑏
)
,
 where 
𝜎
(
𝑧
)
=
1
1
+
𝑒
−
𝑧
p=σ(w⋅x+b), where σ(z)= 
1+e 
−z
 
1
​
 
其中 $p$ 是預測為正類的機率。模型訓練通常通過最大化似然函數等價於最小化交叉熵損失來進行。 邏輯迴歸的優點是計算高效、結果可解釋（權重可解釋各特徵對傾向正類的影響）。同時不需要輸入變量與輸出有線性關係，因為它使用非線性的 sigmoid 函數
finereport.com
。但需要樣本量足夠大，否則基於最大似然的參數估計不穩定
finereport.com
。 應用場合： 常用於醫療診斷（如預測患者是否患病）、營銷（預測用戶是否會購買）、金融風控（預測貸款違約與否）等二元分類任務。也可以經由softmax 推廣用於多元分類。 實作步驟： 使用 Scikit-learn 的 LogisticRegression 可快速建立分類模型。以下示範用乳癌資料集來訓練邏輯迴歸並評估性能：
python
複製
編輯
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 載入乳癌二元分類資料集
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test = X[:500], X[500:]
y_train, y_test = y[:500], y[500:]

clf = LogisticRegression(max_iter=1000)  # 建立模型, 指定較高的最大迭代次數確保收斂
clf.fit(X_train, y_train)               # 模型訓練
y_pred = clf.predict(X_test)            # 模型預測

# 計算性能評估指標
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
print(f"Accuracy={acc:.2f}, Precision={prec:.2f}, Recall={rec:.2f}")
此範例使用 Wisconsin 乳腺癌資料集（數據包含腫瘤的特徵，標籤為良性/惡性），訓練邏輯迴歸模型判斷腫瘤是否惡性。我們分割資料為訓練集和測試集，訓練模型後預測並計算準確率、精確率和召回率等評估指標（這些指標涵義見後文模型評估章節）。範例中將 max_iter 提高至1000以確保優化迭代能收斂。 結果解讀： 輸出的 Accuracy（準確率）為模型預測正確的比例。Precision（精確率）是模型預測為正的樣本中實際為正的比例；Recall（召回率）是實際為正的樣本中被模型正確找出的比例。這些指標能從不同角度評估模型的分類性能。對於不均衡資料集，單看 Accuracy 可能誤導（詳見模型評估章節）。 小技巧與常見錯誤：
特徵標準化： 邏輯迴歸本質是線性模型，加上 sigmoid 激活。在高維時常加入 L2 正則化防止過擬合。使用正則化時務必對特徵進行標準化，否則尺度大的特徵會過度影響結果。
闗聯特徵: 若特徵存在強共線性（相關性高），可能影響參數穩定性。此時可考慮降維（如 PCA）或去除部分共線性特徵。
閾值調整： 預設0.5為分類閾值。在某些應用下（如疾病檢測希望寧可多錯報也不漏報），可適當降低閾值以提高召回率，或反之提高閾值以提高精確率。
常見錯誤： 忽視資料不平衡問題。若一個類別佔絕大多數，模型可能傾向預測所有輸入為該類以獲得高準確率
vocus.cc
。此時需要採用適當的評估指標（如精確率/召回率）或對資料採取重抽樣、調整class_weight等措施改善。
k近鄰演算法 (K-Nearest Neighbors, KNN)
理論講解： K近鄰是一種基於距離的非參數分類和迴歸方法。分類情況下，對於新的輸入樣本，演算法找出訓練集中與其最接近的 K 個鄰居，根據鄰居們的類別以多數決決定該樣本的類別
medium.com
。它體現了「近朱者赤，近墨者黑」的概念：樣本的性質由其鄰近樣本所決定
medium.com
。迴歸問題則取 K 個鄰居目標值的平均作為預測。 KNN 的主要優點是概念簡單、對分佈沒有假設，適合複雜分佈的資料。缺點是計算量大（對每個預測都要計算與大量樣本的距離）且不易解釋模型機制，同時在高維度下效能會急劇下降（所謂維度詛咒）。此外，K值的選擇很關鍵：K 太小模型容易受雜訊干擾，K 太大又可能把距離較遠的點也納入考量反而降低精度。 應用場合： KNN 可用於分類如圖片識別（早期方法）、推薦系統（找相似用戶）等。但由於效率問題，在大規模、高維資料中較少單獨使用，更常作為基準模型或少量資料的簡易模型。 實作步驟： Scikit-learn 的 KNeighborsClassifier 可快速使用 KNN。請確保在使用前已對特徵適當縮放（KNN 基於距離計算，量綱不同的特徵會影響距離計算）。以下以經典的 Iris 資料集訓練一個 KNN 分類器為例：
python
複製
編輯
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 載入 Iris 資料
iris = load_iris()
X, y = iris.data, iris.target

# 分割訓練/測試集並做特徵標準化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 訓練 KNN 分類模型 (K=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)

print("預測的前10個結果:", y_pred[:10])
print("實際的前10個標籤:", y_test[:10].tolist())
print("測試集準確率:", knn.score(X_test_std, y_test))
這裡將鳶尾花資料按7:3切分為訓練和測試集，並對特徵進行標準化。建立 KNN 模型時指定鄰居數 K=5，模型訓練即儲存樣本資料。預測時，輸出部分預測結果與實際標籤進行比較，並計算整體測試準確率（即 score 方法調用）。 結果解讀： 輸出的預測標籤與實際標籤比較可以看到模型分類是否正確。測試集準確率則量化了模型性能，例如若顯示 0.95 表示 95% 的測試樣本被正確分類。 小技巧與常見錯誤：
K值選擇： 一般可透過交叉驗證選擇最佳 K 值。K 通常取奇數以避免票數平手，且不要過小。常試的範圍在 3 到 15 之間。
距離度量： 默認使用歐氏距離，可根據資料性質選擇其他距離（如曼哈頓距離）。對於某些特徵是循環週期性質（如月份、星期幾），需要特別處理距離計算方式。
資料權重： Scikit-learn允許設定鄰居加權，如較近的鄰居權重大（weights='distance'）。在樣本分佈不均時可以改善效果。
常見錯誤： KNN 不具訓練過程，卻需要在預測時保留全部訓練樣本，資料量大時記憶體和時間開銷都很高。實作時若資料量龐大需謹慎使用或考慮近似方法（如 KD-tree 加速或降維）。
支援向量機 (Support Vector Machine, SVM)
理論講解： 支援向量機是一種既可用於分類也可用於迴歸的強大模型。以分類為例，其核心思想是在特徵空間中尋找一個最大間隔的決策超平面來分隔不同類別
zh.wikipedia.org
。線性可分的情況下，SVM 選擇使正負兩類離分界超平面最近的點（支援向量）距離最大化的那條線/面
zh.wikipedia.org
。這保證了分類邊界的魯棒性。對線性不可分資料，SVM 藉由核技巧 (Kernel Trick) 隱式地將原始特徵映射到更高維的空間，使資料在該空間可分
immortalqx.github.io
。常用核函數有高斯 RBF 核、多項式核等。 簡單理解，SVM 嘗試解決兩類問題：「如何將資料點用超平面分開」以及「如何選擇超平面使分隔邊界最寬」。其優點是在適當的核下有強大的分類能力，對高維資料也表現良好，理論基礎完備。缺點是對參數和核的選擇較敏感，對於大型資料集訓練速度較慢，對缺失值也很敏感。 應用場合： SVM 過去在圖像分類、文本分類（如SVM+TF-IDF 用於垃圾郵件檢測）等任務中表現出色。在深度學習興起前，SVM 曾是許多競賽的首選模型之一。目前在中小型且特徵空間較複雜的問題中仍然是有效的方案。 實作步驟： Scikit-learn 的 SVC 實現了支持向量分類。訓練 SVM 時，需關注正則化參數 C（控制間隔寬度與分類錯誤的權衡）和核函數及其參數（如 RBF 核的 gamma）。下面在 Iris 資料集上以 RBF 核訓練一個 SVM 模型：
python
複製
編輯
from sklearn.svm import SVC

# 使用前面的 Iris 資料 (X_train_std, X_test_std, y_train, y_test) 進行 SVM 訓練
svm = SVC(kernel='rbf', C=1.0, gamma='scale')  # RBF 核, 預設 gamma='scale' 自動設定合理值
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)

print("SVM 測試集準確率:", svm.score(X_test_std, y_test))
我們直接利用先前切分並標準化的 Iris 資料訓練 SVM 分類器。這裡選用 RBF kernel（高斯徑向基核）並使用預設的 gamma='scale'（根據特徵數自動計算），C 設為 1.0。模型訓練後，使用 score 方法輸出測試集準確率。 結果解讀： SVM 在這個簡單任務上通常能取得高準確率（接近100%）。更有意義的是理解參數對模型的影響：C 越大，對分類錯誤“零容忍”，間隔可能變窄但錯分更少；gamma 越大，高斯核映射中每個點的影響範圍越小，模型變得更複雜，可能過度擬合。透過調整這些參數，SVM 可在偏差與方差間取得平衡。 小技巧與常見錯誤：
資料縮放： SVM 對特徵尺度非常敏感，必須對輸入進行標準化或正規化，否則某特徵量綱較大會主導距離計算和分類結果。
核函數選擇： RBF 核是較萬用的選擇。如果特徵與目標大致線性，可用線性核（等價於線性SVM）；文本分類常用多項式核或線性核。高斯核的 gamma 需調參，通常與 C 一起以網格搜尋調優。
效率問題： SVM 訓練時間隨樣本量增長近似二次方。對上十萬級樣本可能較慢。此時可考慮採樣資料訓練或使用分塊訓練、線性 SVM 的近似方法（如 SGDClassifier）等。
常見錯誤： 不平衡資料情況下，SVM 仍可能傾向多數類。可透過參數 class_weight='balanced' 讓模型對少數類給予更高權重，以減輕偏態數據的影響。
決策樹 (Decision Tree)
理論講解： 決策樹是一種樹狀的預測模型，使用一連串的if-else 規則將資料逐步細分來做出決策
medium.com
。每個內部節點代表一個特徵測試，分支代表測試結果（特徵值落在哪個區間或類別），葉節點則給出預測結果
medium.com
。決策樹可以用於分類或迴歸。以分類為例，演算法藉由遞迴分割 (recursive partitioning) 構建樹，每次選擇一個特徵及其切分點將資料集劃分，以最大化某種純度提升（如信息增益或基尼指數減少）
medium.com
。 通俗地說，決策樹模擬人類的層層推理過程
medium.com
。例如判斷是否批准貸款，第一步可能問「申請人收入是否高於 X？」根據是或否進入不同的下一問，直到得到結論。決策樹的優點是易於理解（可視化後與人類規則類似）
medium.com
、對資料分佈無過多假設、能處理類別特徵、不需特徵縮放等
ntudac.medium.com
。缺點是單一樹可能容易過度擬合（樹長得太深）且預測可能有階梯效應（迴歸時）。 應用場合： 因其可解釋性，決策樹常用於需要解釋的領域，如信用風險評估、醫療診斷決策等。另外，決策樹是構成集成模型（隨機森林、梯度提升樹）的基本組件，在 Kaggle 比賽等場合大量使用。我們先理解單一決策樹，再在後續章節討論集成方法。 實作步驟： Scikit-learn 的 DecisionTreeClassifier / DecisionTreeRegressor 用於建立決策樹。需要調整的重要參數包括 max_depth（最大樹深）、min_samples_split（進行再分支所需最小樣本數）等，以防止過深的樹過擬合資料。以下在乳癌資料集上訓練一棵分類樹作為示範：
python
複製
編輯
from sklearn.tree import DecisionTreeClassifier, export_text

# 使用前面的乳癌資料集 (X_train, X_test, y_train, y_test)
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print("決策樹準確率:", tree.score(X_test, y_test))

# 輸出決策樹的文字規則
rules = export_text(tree, feature_names=load_breast_cancer().feature_names.tolist())
print("\n決策樹規則：\n", rules)
在這裡，我們限制決策樹最大深度為3，以防止它長得過深過複雜。訓練後計算測試集上的準確率，然後使用 export_text 將決策樹的規則以純文字輸出，方便我們理解樹的決策路徑。 結果解讀： 準確率提供了模型性能的一個指標。而輸出的規則文字（節選）可能類似：
lua
複製
編輯
|--- mean radius <= 14.39
|   |--- worst area <= 548.80
|   |   |--- class: 0 (良性)
|   |--- worst area > 548.80
|   |   |--- class: 1 (惡性)
|--- mean radius > 14.39
|   |--- ... （以下略）
這描述了決策樹的部分決策過程。例如，第一個節點用 mean radius 特徵做切分，一條路徑表示如果半徑不大且 worst area 也不大則預測為良性腫瘤。如此一路向下，葉節點處給出預測類別。這種規則形式非常直觀可解釋
medium.com
。 小技巧與常見錯誤：
避免過度擬合： 調參限制樹複雜度很重要。除了 max_depth 外，也可設定 min_samples_leaf（葉節點最少樣本數）或 max_leaf_nodes（最多葉節點數）。一個常見策略是先生成一棵完全生長的樹，再透過**剪枝 (pruning)**策略減少其複雜度。
資料不平衡處理： 若分類樹遇到不平衡資料，可能偏向主流類別。可在劃分指標中加入權重或在建立樹前進行上採樣/下採樣處理。
常見錯誤： 使用決策樹時忘記處理類別型變數的編碼。Sklearn 的決策樹實現需要將類別特徵數值化（可以使用獨熱編碼或直接用數值表示類別，因為決策樹基於排序不會誤用類別編碼的大小關係）。
優化注意： 不要奢望找到全局最優的決策樹——決策樹演算法使用貪婪策略局部最佳分割，不一定是全局最優，但效果通常已足夠好且計算可接受。
集成學習：隨機森林與梯度提升 (Ensemble Learning: Random Forest & Boosting)
單一模型有時預測能力有限，集成學習 (Ensemble Learning) 透過結合多個模型來提升預測性能
zhuanlan.zhihu.com
。“三個臭皮匠，勝過諸葛亮”，多個弱模型適當結合往往能形成一個強模型
zhuanlan.zhihu.com
。集成方法主要分為兩類：
tomohiroliu22.medium.com
裝袋法 (Bagging)：如隨機森林，透過對資料採樣訓練多棵獨立的樹模型，最後平均預測結果
ntudac.medium.com
。藉由多樣性來降低模型方差，提升穩健性。
提升法 (Boosting)：如 AdaBoost、梯度提升樹 (GBDT)、XGBoost 等，序列地訓練弱模型，每個新模型針對前一個模型的錯誤進行改進
books.com.tw
。最終將所有弱模型加權相加，降低預測偏差。
隨機森林 (Random Forest)
理論講解： 隨機森林由許多獨立的決策樹構成。每棵樹在訓練時會從原始資料隨機取樣（bootstrap sample）並在每個節點隨機選擇部分特徵考慮分裂。這種雙隨機引入差異，使樹與樹之間預測誤差不相關。最後用所有樹預測的眾數（分類)或平均（迴歸）作為結果
ntudac.medium.com
。隨機森林通常具有高準確率、不易過擬合（由於取平均降低方差）、能評估特徵重要性等優點。其缺點是失去單棵樹的可解釋性，且預測速度變慢（因需要匯總多樹結果）。 實作步驟： Scikit-learn 的 RandomForestClassifier 與 RandomForestRegressor 對應分類和迴歸。主要參數有樹的數量 n_estimators、最大深度、max_features（每次分裂考慮的隨機特徵數）等。一般情況下樹數多一些可以增進效果但也增加計算量。範例：
python
複製
編輯
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 延續先前乳癌資料集 (X_train, X_test, y_train, y_test)
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("隨機森林準確率:", accuracy_score(y_test, y_pred))

# 顯示特徵重要性
importances = rf.feature_importances_
top_features = sorted(zip(load_breast_cancer().feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
print("Top5 特徵重要性:", top_features)
此程式建立100棵樹的隨機森林分類器（允許樹生長到最大深度）。模型訓練後計算測試集準確率，並輸出隨機森林估計的 Top5 特徵重要性。隨機森林會計算每個特徵在所有樹分裂中減少不純度的總和作為重要性指標。 結果解讀： 隨機森林的準確率通常比單顆決策樹要高，而且對訓練集和測試集的表現接近（避免過擬合）。特徵重要性提供了模型判斷的依據，例如輸出可能顯示 worst radius、worst texture 等特徵權重較高，意味著這些特徵對模型決策貢獻較大。 小技巧與常見錯誤：
樹數與收斂： 樹的數量足夠多時，森林性能趨於穩定並不再顯著提升，過多的樹只是增加計算。可透過觀察OOB誤差（袋外樣本誤差）曲線找到收斂點。
參數調整： 一般隨機森林對參數不太敏感，默認值已不錯。可適當調整 max_features：較小值增加多樣性降低方差，較大值減少偏差但樹間相關性提高。
常見錯誤： 切忌將隨機森林當作完全黑盒。雖然單棵樹不可解釋性下降，但透過特徵重要度或部分依賴圖仍可一定程度了解模型。忽視這點在敏感領域可能導致難以解釋的結果。
梯度提升樹 (Gradient Boosting Trees)
理論講解： 提升法通過序列地組合多個弱模型，使後一個模型補前一個模型之不足。梯度提升是一種提升法框架：每一步訓練一棵新樹來擬合當前模型的殘差（即真實值與目前模型預測值之差），通過逐步減少殘差來逼近真實函數。這相當於在函數空間執行梯度下降，每一步沿著梯度方向（殘差）訓練新的樹模型
books.com.tw
。常見實現如XGBoost、LightGBM 基於此思想進行了工程上的優化。 相比隨機森林並行地平均結果，提升法是累加弱學習器的效果，因此能取得更低的偏差，但也更容易過擬合，需要嚴格的正則化（如限制每棵子樹深度、採用子採樣、加入學習率等）。優點是在充足正則下往往能取得極高的預測準確率；缺點是參數較多需要調整，訓練過程序列化難以並行。 實作步驟： Scikit-learn 提供 GradientBoostingClassifier（簡化版，不及 XGBoost 強大但原理一致）。關鍵參數有 n_estimators（弱學習器數目）、learning_rate（學習率，降低每棵樹貢獻以需要更多迭代但減少過擬合風險）以及每棵子樹的複雜度控制如 max_depth。簡單示例：
python
複製
編輯
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbc.fit(X_train, y_train)
print("梯度提升樹準確率:", gbc.score(X_test, y_test))
此處我們訓練100棵深度為3的子樹，學習率0.1（較典型的默認值）。在實際應用中，常需要用較小的學習率（如0.01），並增大樹的數量，配合交叉驗證尋找最佳點。 結果解讀： 梯度提升樹通常能取得與隨機森林相當或更高的準確率，特別是在資料量適中且特徵資訊豐富時。若發現訓練集分數明顯高於測試集，表示過擬合，應降低樹深或學習率等。 小技巧與常見錯誤：
*學習率與樹數: * 兩者有trade-off。較小學習率需要更多樹才能訓練充分，訓練時間變長；較大學習率則可能較快擬合但風險高。一般固定一個調整另一個，常見做法是將學習率調低（如0.05以下）然後選擇足夠多的樹。
*正則化: * 梯度提升有多種正則化手段，除上述的max_depth、子抽樣(subsample每次用部分樣本訓練)、min_samples_leaf等。一些庫如XGBoost還提供reg_alpha、reg_lambda L1/L2正則化參數。
常見錯誤： 忽略缺失值處理。決策樹類模型可天然處理缺失值（一些實作甚至有缺失值分支機制），但最好在訓練前統一定義缺失處理策略，否則不同實作處理方式不同會影響模型結果。
*計算資源: * Boosting無法像bagging那樣輕易並行，每一步依賴前一步結果。大型數據上使用需要注意訓練時間，可考慮分佈式實作（如XGBoost的分散版本）。
非監督式學習 (Unsupervised Learning)
非監督式學習處理無標記資料，模型試圖從資料中發掘內在的結構或模式
medium.com
。沒有標準答案的指引，結果需靠人解讀，因此難度較高
medium.com
。常見非監督任務包括分群 (Clustering)、關聯規則學習 (Association Rule Learning) 和降維 (Dimensionality Reduction) 等
ibm.com
weilihmen.medium.com
。本節將介紹幾種主要的非監督技術及其應用。
分群 (Clustering)
理論講解： 分群旨在將資料集中的樣本劃分為若干組，使同群內的樣本彼此相似，不同群的樣本差異明顯
medium.com
。這是一種探索性分析，典型演算法有 K-Means、階層式分群、DBSCAN 等。其中 K-Means 是最常用的劃分式分群演算法：它假設群集的形狀為球狀，由中心（質心）定義，透過迭代更新質心位置分配樣本，使組內誤差平方和最小。K-Means 需要事先指定群集數目 K，並對初始值敏感（通常多次隨機初始化取最佳結果）。 應用場合： 分群可用於資料探索和客群劃分。例如顧客分群可將用戶按行為分為數類，以便行銷差異化策略；圖像分群可將未標記的圖像按內容相似度分組。需要注意，分群結果沒有絕對對錯，實用價值取決於人們對群組的解讀。 實作步驟： 使用 Scikit-learn 的 KMeans 來示範分群。下面生成一組二維資料點，然後用 K-Means 分為 3 群，並視覺化結果：
python
複製
編輯
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 生成模擬數據：3個群，每群中心分別在 (1,1), (5,5), (9,1)
np.random.seed(42)
cluster1 = np.random.randn(50, 2) + np.array([1, 1])
cluster2 = np.random.randn(50, 2) + np.array([5, 5])
cluster3 = np.random.randn(50, 2) + np.array([9, 1])
X = np.vstack([cluster1, cluster2, cluster3])

# 執行 K-Means 分群
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# 繪製分群結果
for label in np.unique(labels):
    plt.scatter(X[labels==label, 0], X[labels==label, 1], label=f'Cluster {label}')
plt.scatter(centroids[:,0], centroids[:,1], s=100, c='black', marker='X', label='Centroids')
plt.legend()
plt.title('K-Means Clustering Result')
plt.show()
medium.com
medium.com
上圖顯示了分群的結果，彩色點代表不同群組，黑色叉叉為每群的中心點。可見 K-Means 成功將資料區分為3組，且中心點大致位於模擬時設定的中心附近。此圖說明瞭 K-Means 的聚類效果及聚類中心的意義。 結果解讀： 分群沒有“正確答案”，但我們可依據 domain knowledge 評估有無意義。例如這3群資料本就是我們生成的，K-Means 恢復了這些分布。實際應用中，可根據群內樣本的共通特徵來賦予每群業務上意義，例如群0為“高價值客戶群”、群1為“潛在流失客戶群”等（需要額外分析）。 小技巧與常見錯誤：
選擇 K 值： K-Means 需決定群數K，可用**肘部法則 (Elbow method)觀察總體內距離和隨K變化的趨勢，或輪廓係數 (Silhouette score)**等指標評估不同K的聚類質量，再選擇合理值。
標準化輸入： 與距離有關的聚類對特徵尺度敏感，所以通常要標準化特徵再進行聚類，以免一兩個量綱特別大的特徵主導距離計算。
初始化影響： K-Means 可能收斂到局部最優解。Scikit-learn 的實現透過多次初始化 (n_init) 緩解這一問題。可以增加初始化次數或嘗試 k-means++ 初始化（默認即是）取得更穩定結果。
常見錯誤： 誤用分群結果。因非監督學習沒有標籤可驗證，聚類結果需謹慎解讀。例如將分群結果當作分類模型來對新數據預測可能不恰當，因為聚類更多是探索性而非預測性任務。
關聯規則學習 (Association Rule Learning)
理論講解： 關聯規則學習尋找資料中不同項目之間的共現關係。例如著名的“啤酒與尿布”案例就源自關聯分析
weilihmen.medium.com
：從購物交易資料中發現經常一起購買的商品組合。核心概念有：
頻繁項集： 出現頻率高於某閾值的項目集合。
關聯規則： 形式為 $X \implies Y$，表示若包含項集 $X$，則往往也包含 $Y$。
支持度 (Support)：$P(X \cup Y)$，即某項集在資料中出現的比率
weilihmen.medium.com
。
信賴度 (Confidence)：$P(Y|X)$，在含有 $X$ 的情況下也含 $Y$ 的比例
weilihmen.medium.com
。
提升度 (Lift)：$\frac{P(Y|X)}{P(Y)}$，表示有 $X$ 時同時有 $Y$ 的機率是沒有 $X$ 時的幾倍
weilihmen.medium.com
。
舉例：“尿布 ⇒ 啤酒” 規則的支持度是交易中同時買尿布和啤酒的比率，信賴度是買了尿布的顧客中有多少也買啤酒，提升度則衡量這種關聯性是否超出隨機巧合
weilihmen.medium.com
。Apriori 演算法是經典的挖掘頻繁項集的方法，它利用反單調性原理：若一個集合不頻繁，則其任一超集也不會頻繁
weilihmen.medium.com
。這使得演算法能夠有效剪枝搜索空間。 應用場合： 關聯規則主要應用在購物籃分析（市場銷售數據）、推薦系統（根據用戶行為推薦相關物品）等。例如電商網站利用關聯規則發現“經常一起購買”的商品，據此產生推薦。 實作步驟： Python 有如 mlxtend 等庫可以用 Apriori 挖掘頻繁項集並產生關聯規則。下面示範一個小型交易資料的關聯分析：
python
複製
編輯
!pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

# 模擬交易記錄，每列為一筆交易中購買的商品（A,B,C表示不同商品）
transactions = [
    {'A': 1, 'B': 1, 'C': 0},
    {'A': 1, 'B': 1, 'C': 1},
    {'A': 0, 'B': 1, 'C': 1},
    {'A': 1, 'B': 0, 'C': 1},
]
df_transactions = pd.DataFrame(transactions)
# 使用 apriori 找出支持度>=50%的頻繁項集
freq_itemsets = apriori(df_transactions, min_support=0.5, use_colnames=True)
print("頻繁項集:\n", freq_itemsets)
# 推導關聯規則，最小信賴度 0.7
rules = association_rules(freq_itemsets, metric="confidence", min_threshold=0.7)
print("\n關聯規則:\n", rules[['antecedents','consequents','support','confidence','lift']])
在這簡單例子中，我們構造了4筆交易資料，其中商品 A、B 出現頻率較高。我們使用 mlxtend 的 apriori 函式找出支持度不低於50%的頻繁項集，再透過 association_rules 求出信賴度不低於0.7的關聯規則。輸出顯示頻繁項集及對應的關聯規則，例如可能發現 {A,B} 是頻繁項集，且規則 A⇒B、B⇒A 具有高信賴度和提升度（具體取決於模擬資料）。 結果解讀： 頻繁項集列表展示了高支持度的項目組合，例如 {A, B} 支持度 0.75 表示75%的交易都同時包含 A 和 B。關聯規則部分則輸出如 frozenset({'A'}) -> frozenset({'B'}) 信賴度0.8提升度1.07，表示有A時買B的機率是總體買B機率的1.07倍，關聯性不算很強。但在真實場景中，若某規則提升度遠高於1，則代表強關聯關係，可作為商業決策參考。 小技巧與常見錯誤：
閾值選擇： 支持度和信賴度閾值的選定需平衡結果數量與有意義程度。支持度太低會產生海量規則，多數可能僅是偶然巧合；太高又可能漏掉有價值的長尾模式。
提升度解讀： 提升度 > 1 表示正相關關係，< 1 表示負相關（有X時反而較少有Y）。在推薦系統中，可選取高提升度的規則用於關聯推薦。
常見錯誤： **規則迷信：**關聯規則是相關性非因果性，不能直接推論因果關係。例如“尿布⇒啤酒”不代表買尿布會導致想買啤酒，可能背後有隱含因素（年輕父親的行為模式）。應謹慎運用規則進行商業決策，通常需結合領域專家的判斷。
計算複雜度： Apriori 在大型資料上計算成本高（組合爆炸）。可考慮 FP-growth 等更優演算法或降低閾值要求。如果資料過大，也可能需要採樣或使用分散式計算。
主成分分析 (PCA) – 非監督式降維
(PCA 已於前面特徵工程部分介紹，在此不再重複。如需，可在此處補充非監督視角對 PCA 的解釋。) 理論講解： 如前所述，主成分分析是一種常用的降維技術。站在非監督的角度，PCA 嘗試尋找數據在沒有標籤指引下的最大變異方向，認為變異大的方向可能包含更多信息
medium.com
。透過這種方式壓縮資料維度，可以在損失少量信息的前提下大幅減少特徵數量，是非監督學習中資料探索和可視化的利器。 應用場合： 資料可視化（將高維資料投影到2D/3D空間觀察聚類情況）、資料預處理（加速後續監督模型的訓練）等。 實作步驟與結果解讀：（略，已在前述特徵工程部分有範例。） 小技巧與常見錯誤：
主成分選擇： 可透過累積解釋變異比例選擇保留多少主成分，如選擇能解釋85%以上變異的前N個主成分，以平衡降維後信息保留。
資料中心化： 在 PCA 前通常要對資料中心化（每個特徵減去均值），否則第一主成分可能會對應均值而非變異方向。此外若特徵量綱差異大，也需標準化以防止大尺度特徵主導主成分。
常見錯誤： 將PCA用在有意義的分類變數上並不適當，因為PCA假設連續數值空間中的變異最大方向才有意義，對純類別變數或One-Hot編碼資料，可能降維效果不佳且不易解讀（可考慮用TSNE或UMAP等替代）。
增強式學習 (Reinforcement Learning)
理論講解： 增強式學習是一種不同於監督/非監督的學習範式。智能體 (Agent) 在環境 (Environment) 中試探性地執行行動 (Action)，環境會根據行動給予獎勵 (Reward)或懲罰，智能體的目標是在長期累積獎勵最大化的前提下學習最佳策略
ai4dt.wordpress.com
。特點是沒有直接的正確行為指引，只有延遲的獎勵信號，智能體需自己探索出好的行為。 增強學習常用馬可夫決策過程 (MDP) 作建模，在每個狀態下，不同行動有不同的即時獎勵及未來回報。經典演算法如 Q-learning 透過不斷更新 Q 函數（狀態-行動的價值估計）來逼近最優策略。策略梯度方法則直接優化策略函數。探索 (Explore) 與利用 (Exploit) 的平衡是增強學習的核心挑戰：智能體需要探索新的行為以發現更好策略，同時逐漸傾向利用已知優秀策略以累積獎勵
ai4dt.wordpress.com
。 應用場合： 增強學習在遊戲 AI（AlphaGo 下棋、自動玩電動遊戲）、機器人控制、自動駕駛（決策控制部分）等領域有著重要應用。例如 AlphaGo 就是透過與自己對弈（環境為棋局狀態，行動為下棋步，獎勵為勝負結果）並不斷強化策略而達到超人水準。 實作方法： 增強學習的實作通常需模擬環境。OpenAI Gym 是常用的工具包，提供如經典控制、遊戲等環境。這裡以 pseudo-code 描述 Q-learning 基本流程：
pseudo
複製
編輯
Initialize Q(s, a) arbitrarily for all state-action pairs
for each episode:
    initialize state s
    for each step in episode:
        以 epsilon-greedy 選擇動作 a (以 epsilon 機率隨機，1-epsilon 機率選 Q 最大)
        執行動作 a，觀察獲得獎勵 r 和下一狀態 s'
        更新： Q(s, a) ← Q(s, a) + α * [r + γ * max_{a'} Q(s', a') - Q(s, a)]
        s ← s'
        若 s' 為終止狀態，則跳出
其中 α 是學習率，γ 是折扣因子。這個演算法讓 Q 值逐步逼近最優 Q函數。待 Q學習收斂後，對任何狀態選擇 Q 值最大的動作即為最優策略。 常見技巧： 在實踐中，會使用經驗回放（存儲過去經驗隨機抽取訓練，降低樣本相關性）、深度神經網絡 近似 Q 函數（形成 DQN 演算法），或使用策略梯度及Actor-Critic等進階方法來應對連續動作空間等更複雜情況。 小技巧與常見錯誤：
獎勵設計： 增強學習中，如何設計合理的獎勵函數非常關鍵。一個不當的獎勵可能導致智能體學到偏差甚至荒謬的策略（例如為了得到獎勵而鑽系統漏洞）。獎勵需清晰指引目標且平衡短期/長期利益。
探索策略： ε-greedy 是簡單常用的探索策略，但在高維連續空間效果不好。可考慮更進階策略或隨時間衰減 ε 值來逐步從探索轉向利用。
常見錯誤： 訓練環境和實際部署環境不一致。如果智能體在模擬環境學到策略但真實環境動態不同，直接應用可能失敗，需要域隙遷移或在真實環境進行額外訓練。此外，增強學習的隨機性較大，每次訓練結果可能不同，需要足夠多試驗和良好的參數調整。
模型評估與調優 (Model Evaluation & Tuning)
建立模型後，需要衡量其在看不見資料上的表現，並可能進一步調整優化。本章討論模型評估的指標與方法，以及如何進行模型超參數調整和比較不同模型。
評估指標與混淆矩陣
分類模型評估： 對於分類問題，混淆矩陣 (Confusion Matrix) 是基本的性能刻畫工具
vocus.cc
。以二元分類為例，混淆矩陣包含四種情況：
真正類 (TP)：實際正類，被模型預測為正類。
假正類 (FP)：實際負類，但模型誤預測為正類（Type I error）。
假負類 (FN)：實際正類，但模型預測為負類（Type II error）。
真負類 (TN)：實際負類，模型預測為負類。
從這四格數據，可計算多種評估指標：
準確率 (Accuracy) = (TP+TN)/(TP+FP+FN+TN)，即模型預測正確的比例。這是最直觀的指標，但在資料不平衡時可能失真
vocus.cc
。例如，有990個負類、10個正類，如果模型全預測為負類，準確率達99%但完全忽略正類。
精確率 (Precision) = TP/(TP+FP)，預測為正的案例中有多少是真的正。精確率低意味著誤報多。
召回率 (Recall) = TP/(TP+FN)，實際為正的案例中有多少被預測出。召回率低意味著漏檢多。
F1-score = 2 * Precision * Recall / (Precision + Recall)，精確率和召回率的調和平均，在兩者都重要時使用。
特異度 (Specificity) = TN/(TN+FP)，針對負類的召回率。
ROC曲線 (Receiver Operating Characteristic)：描繪分類器在各種閾值下的性能曲線，以真正率 (TPR) 對 假正率 (FPR)，即 Y 軸為 Recall，X 軸為 1-Specificity
medium.com
。ROC 曲線靠近左上方表示性能好
medium.com
。
AUC (Area Under Curve)：ROC曲線下的面積
medium.com
。AUC=1表示完美分類，0.5表示無判別資訊的隨機分類器。
迴歸模型評估： 常用均方誤差 (MSE)、均方根誤差 (RMSE)、平均絕對誤差 (MAE)、決定係數 $R^2$ 等。MSE 對離群值較敏感，MAE 更健壯；$R^2$ 表示模型解釋變異的比例。 實作方法： Scikit-learn 提供了 metrics 模組計算上述指標。我們已在前例中示範過 Accuracy、Precision、Recall 的使用。以下展示如何計算混淆矩陣與 ROC-AUC：
python
複製
編輯
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# 以之前邏輯迴歸的預測結果 y_pred 和真實標籤 y_test
cm = confusion_matrix(y_test, y_pred)
print("混淆矩陣:\n", cm)

# 若需要 ROC-AUC, 需獲得陽性類的概率分數
y_scores = clf.predict_proba(X_test)[:,1]  # 預測為正類的機率
auc = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
print("測試集 ROC-AUC:", auc)
上述代碼先計算混淆矩陣，輸出類似：
lua
複製
編輯
[[TN, FP],
 [FN, TP]]
然後計算 ROC-AUC 分數以及 ROC 曲線的座標數據（fpr, tpr）。AUC 值提供了分類器對正負類判別能力的整體評價，不受選擇閾值影響。 結果解讀： 例如混淆矩陣 [[50, 5], [3, 42]] 表示 TN=50, FP=5, FN=3, TP=42。精確率 = 42/(42+5)=89.4%，召回率 = 42/(42+3)=93.3%。AUC 值若接近1表示模型區分能力很好，接近0.5則幾乎無區分能力。 小技巧與常見錯誤：
不平衡資料評估： Accuracy 在不平衡資料下意義不大，此時 Precision/Recall/F1 更能反映模型對少數類的照顧程度
vocus.cc
。PR 曲線（Precision-Recall curve）也是評估不平衡分類器的有用工具。
多類評估： 對多分類問題，可透過宏平均 (macro-average) 或加權平均 (weighted-average) 的 Precision/Recall/F1 進行整體評估，或計算每類的指標。
閾值調整： 根據 ROC 曲線或業務需求選擇分類閾值，可以視情況偏重 Precision 或 Recall。例如疾病篩檢寧可多報（高Recall）也不漏掉真陽性，但垃圾郵件過濾則寧可漏掉也不錯殺正常郵件（高Precision）。
常見錯誤： 只關注單一指標。模型性能需多維度考察，例如高Precision低Recall可能不實用，必須結合任務要求。還有不要忘記進行統計顯著性檢驗（如confidence interval或t-test）特別是在比較兩模型優劣時，以確認差異不是隨機波動。
交叉驗證 (Cross-Validation)
理論概念： 評估模型時，我們通常將資料分為訓練集和測試集。然而單一次切分可能導致評估結果對資料劃分方式較敏感（尤其資料量不大時）。交叉驗證通過多次隨機劃分評估取平均，獲得更穩健的模型表現估計
ntudac.medium.com
。 最常用的是 K 折交叉驗證 (K-fold CV)
ntudac.medium.com
：將資料平均分成 K 份，進行 K 輪實驗。每輪選擇其中一份作為驗證集，剩餘 K-1 份作訓練集，評估模型在該驗證集的表現。最終得到 K 個評估分數取平均作為模型效能指標
ntudac.medium.com
。這確保每個樣本都恰好被驗證一次、訓練 K-1 次。常用 K=5 或 10。 交叉驗證不僅用於模型最終性能估計，也廣泛用於超參數調優，通過在訓練集內部再劃分驗證來選擇最佳參數（見下一節）。 實作方法： Scikit-learn 的 model_selection.cross_val_score 可以直接對給定模型和資料進行 K 折驗證。示例：
python
複製
編輯
from sklearn.model_selection import cross_val_score

# 對LogisticRegression模型進行5折交叉驗證評估Accuracy
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print("5 折 CV 準確率:", scores)
print("平均準確率: %.3f (+/- %.3f)" % (scores.mean(), scores.std()*2))
這段代碼對之前訓練的邏輯迴歸模型在整個資料集上做5折交叉驗證，計算每折的準確率。輸出如 array([0.94, 0.96, 0.93, 0.95, 0.94]) 及平均值和標準差範圍。 結果解讀： CV 分數列出模型在不同切分下的性能。若每折結果差異較小，表示模型表現穩定；若差異較大，可能模型對資料切分較敏感，或資料集有異質性。平均準確率提供了較單一切分更可靠的性能估計。95%置信區間可用 mean ± 1.96*std/√(n) 計算。 小技巧與常見錯誤：
資料洩漏： 交叉驗證要確保驗證集是模型完全未見過的。例如資料標準化應在每個fold的訓練子集上fit再transform驗證子集，切勿整體fit導致資訊洩漏。同樣地，像特徵選擇等流程都應嵌入CV內進行。
時序資料CV： 對時序相關的資料，不宜隨機CV，應採用時間序列分割（如前80%訓練，後20%驗證，或滑動窗口法）以避免未來數據洩漏給過去。Sklearn 提供 TimeSeriesSplit。
計算成本： K折CV需要訓練K次模型，資料量大或模型訓練慢時成本不低。可考慮折數調小或使用並行運算 (cross_val_score 支援 n_jobs 多核並行)。
常見錯誤： 在整個資料上做過某種處理（如用全部資料調參或特徵篩選）然後再CV，這其實讓測試fold資訊洩漏到了訓練過程，導致評估偏樂觀。正確做法是在CV過程的每個fold內完全獨立地完成訓練、調參等步驟。
超參數調整與模型選擇
理論概念： 模型的超參數（Hyperparameter）是指那些不通過訓練自動學習，而需人工設定的參數，例如決策樹的最大深度、正則化強度、KNN的鄰居數K等。在模型訓練前，需要通過調參找到較優的超參數組合。調參不當可能導致模型欠擬合或過擬合。 常用的超參數調整方法：
網格搜尋 (Grid Search)：設定每個超參數的一組可能取值，窮舉組合進行交叉驗證，選擇表現最佳者
ntudac.medium.com
。優點是能找到全局最優組合（假設搜索空間覆蓋），缺點是組合過多時計算昂貴。
隨機搜尋 (Random Search)：在參數空間中隨機取一定次數組合評估。一般在給定試驗次數下效率高於網格搜尋，因為某些參數對結果影響不大，隨機採樣更為經濟。
貝葉斯優化、進化演算法等：更高級的方法，根據已嘗試結果不斷調整下一組參數選擇，向最佳區域靠攏，適用於需加速高成本調參的情況。
實作方法： Scikit-learn 提供 GridSearchCV 和 RandomizedSearchCV 來自動調參。這些工具會執行嵌套的交叉驗證，避免資訊洩漏。下面以隨機森林為例，用網格搜尋調參：
python
複製
編輯
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("最佳參數組合:", grid_search.best_params_)
print("對應驗證集準確率:", grid_search.best_score_)
這段代碼定義了一個參數網格，包括樹數（50,100,200），最大深度（無限制,5,10），每次分裂考慮特徵數（平方根或對數2）。GridSearchCV 會對每種組合做3折CV計算平均準確率，best_params_ 給出最佳組合，如 {'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 100}，以及相應的平均準確率。模型經調參後，可用 grid_search.best_estimator_ 取得最佳配置的模型，並用全訓練集重訓（GridSearchCV已自動在fit結束時用最佳參數重訓在整個訓練集）。 模型選擇： 在調參的同時，我們也可能比較不同模型。這可以通過在 grid search 的 estimator 列表中放入不同模型實例來一起比較，或者更直接地，分別對多個候選模型各自CV評估性能指標，根據結果選擇最佳模型。 小技巧與常見錯誤：
*搜尋範圍: * 先以較粗的範圍定位較好區域，再在此區域細化搜索（例如先確定樹數量級，再細調）。對連續值超參數可先log尺度試幾點再微調。
避免過擬合調參: 調參本身也可能過擬合驗證集（尤其參數很多時）。可留出一個獨立的測試集最終評估，或在調參過程中使用多重CV嵌套、添加懲罰（如對非常複雜的參數組合適當Regularization）。
計算並行: GridSearchCV 和 RandomizedSearchCV 支援設定 n_jobs 來多核運算（如 n_jobs=-1 用盡可能多CPU），大幅加速調參。
常見錯誤： 忘記對整個流程做交叉驗證。理論上，調參本身最好也在CV框架內進行（所幸Sklearn工具已處理）。千萬不要用測試集指導調參——那等於泄漏了測試資訊，使最後評估結果偏樂觀且不可靠。一旦用測試集調了參，這個測試集就不能再用來作為最終評估，需另備一組未見資料作測試。
模型部署與持續監控
(競賽環境中或許不涉及實際部署，但若需要可提及) 完成模型訓練和評估後，在實務中最後一步是部署模型並對其進行持續監控。部署時通常將模型序列化保存（如使用 pickle 或 joblib 保存 sklearn 模型），在應用程式中載入模型對新資料進行預測。需要注意輸入資料格式需與訓練時一致，包括特徵順序和任何前處理操作。部署後要監控模型表現，一旦發現輸入數據分佈漂移或模型性能下降，需考慮重新訓練或調整模型（這在競賽中體現為線下上線分數差異）。 常見錯誤： 部署環境與開發環境不一致導致模型無法正常運行；忘記應用與訓練相同的資料前處理步驟；缺乏性能監控導致模型失效仍未被察覺。
以上指南涵蓋了 AI 比賽所需知識的方方面面：從編程基礎、資料處理到各類機器學習模型的理論與實作，最後到模型評估與調優。透過本指南的學習，您將在理論理解和實踐操作上都打下堅實基礎。在實戰中，建議結合實際問題多加練習，閱讀經典案例和競賽解題分享，不斷總結經驗。祝您在即將到來的人工智慧比賽中取得佳績！ 參考文獻：
機器學習理論與實務書籍，如《Pattern Recognition and Machine Learning》, 《Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow》等。
Scikit-learn 官方文件與範例
ntudac.medium.com
ntudac.medium.com
、TensorFlow/PyTorch 官方教程等資源。
各類線上課程（Coursera、Udacity）和 Kaggle 學習賽資料，用於進一步提升實戰能力。



























MOAI 2025 國際人工智慧奧林匹克競賽 知識點手冊與範例題解
本手冊彙整 MOAI 2025 澳門選拔賽所需的重點知識，包括 Python 編程基礎、數據處理、資料視覺化、機器學習、模型評估、深度學習 (PyTorch)、卷積神經網絡 (CNN)、自然語言處理 (NLP) 等範疇，並針對提供的手寫數字識別範例題進行詳細解題。內容涵蓋理論定義、常用語法與範例、常見錯誤與調試提示，以及應用場景與最佳實踐。此手冊可作為半開卷比賽的高效查詢資料，幫助參賽者快速復習和定位所需知識。
Python 基礎 (Python Basics)
基本語法與結構： Python 使用縮排來劃分程式區塊，常見結構包括 條件語句 (if/elif/else)、迴圈 (for 迴圈, while 迴圈) 等。注意冒號:和縮排層級。範例：
python
複製
編輯
x = 10
if x > 0:
    print("正數")
else:
    print("非正數")
# 輸出: 正數
常見錯誤 (Common Errors): 忘記在條件後加冒號、縮排錯誤 (IndentationError)、誤用條件運算符 (= vs ==) 等。調試時可檢查括號配對和縮排空格數是否一致。
循環 (Loops)： 迴圈允許重複執行程式區塊。for 迴圈常用於遍歷序列，例如 for i in range(5): 會從0迭代到4。while 迴圈在條件為真時反覆執行。範例：
python
複製
編輯
# 用 for 迴圈計算 1+2+...+N
total = 0
for i in range(1, N+1):
    total += i

# 用 while 迴圈列印列表元素
i = 0
while i < len(my_list):
    print(my_list[i])
    i += 1
常見錯誤： for 迴圈使用錯誤的範圍（如 range(N) 從0到N-1，而非到N）、while 迴圈條件未正確更新導致無限迴圈。最佳實踐是在可能無限的迴圈中加入退出條件，並善用 Python 迭代器或列表生成式提高簡潔性。
函數 (Functions)： 使用 def 關鍵字定義函數以實現模組化程式。例如：
python
複製
編輯
def factorial(n):
    """計算 n 的階乘"""
    result = 1
    for i in range(2, n+1):
        result *= i
    return result

print(factorial(5))  # 輸出: 120
關鍵定義： 函數可以有引數 (parameters)和返回值 (return)，沒有明確 return 的函數會回傳 None。可用 默認參數 和 關鍵字參數 增強靈活性。常見錯誤： 忘記呼叫函數時加括號 (function vs function())、在函數內未使用全域變數宣告就修改全域變數 (需要使用 global 關鍵字) 等。
文件讀寫 (File I/O)： Python 提供簡潔的文件操作介面。使用內建函數 open() 讀寫文件，模式 "r" 讀取、"w" 覆寫寫入、"a" 追加。範例：
python
複製
編輯
# 將清單寫入檔案
data = ["Alice", "Bob", "Charlie"]
with open("names.txt", "w", encoding="utf-8") as f:
    for name in data:
        f.write(name + "\n")

# 讀取檔案內容
with open("names.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    print(lines)
使用 with 語句可自動安全地關閉文件。常見錯誤： 檔案路徑錯誤導致 FileNotFoundError、讀寫時忘記設定正確的編碼導致亂碼。調試時可用 print() 輸出確認路徑，或使用 os.path 函數處理路徑。
第三方庫導入與使用： 利用 import 導入第三方庫，如 import numpy as np 或 from math import pi。確保已安裝對應庫（在 Kaggle 等平台使用 !pip install 安裝）。使用別名能簡化調用（如 np 代表 numpy）。常見錯誤： 模組未安裝或名稱拼寫錯誤 (ModuleNotFoundError)，或誤用 import * 造成名稱衝突。最佳實踐是僅導入所需對象並使用簡潔別名，保持代碼清晰。
數據處理與操作 (Data Handling with NumPy & Pandas)
NumPy 陣列 (ndarray)： NumPy 提供高效的多維陣列，支援向量化運算。創建方式包括 np.array() 從 Python 結構轉換，以及 np.zeros(), np.ones(), np.arange(), np.linspace() 等函數產生陣列。範例：
python
複製
編輯
import numpy as np
a = np.array([1, 2, 3])              # 1維陣列 [1,2,3]
b = np.zeros((2,3))                  # 2x3 全零矩陣
c = np.linspace(0, 1, num=5)         # 等差序列 [0. ,0.25,0.5,0.75,1. ]
陣列操作： 可通過 shape 屬性取得尺寸，使用 reshape 改變形狀；利用切片 (array[start:stop:step]) 或布林索引篩選元素；陣列運算支援逐元素加減乘除，不需顯式迴圈。常見錯誤： 進行不同形狀的陣列運算時若維度不兼容導致 ValueError，此時需要檢查陣列形狀或利用 NumPy 廣播機制（自動擴充維度）調整。例如 a + b 要求 a.shape == b.shape 或其中一方是可廣播的維度。
Pandas 資料框 (DataFrame)： Pandas 提供 DataFrame 用於表格數據處理。可使用 pd.read_csv()、pd.read_excel() 等輕鬆讀取資料。範例：
python
複製
編輯
import pandas as pd
df = pd.read_csv("data.csv")          # 從CSV讀取成DataFrame
print(df.head(3))                    # 查看前3筆資料
print(df.describe())                 # 基本統計描述
資料框操作： 透過 df.shape 獲取維度，df.columns 檢視欄位。選取資料： df["col"] 獲取單欄位序列，df.iloc[行索引, 列索引] 或 df.loc[行標籤, 列標籤] 可按位置或標籤切片。可使用 布林索引 過濾，例如 df[df["age"] > 30] 返回年齡大於30的資料子集。資料轉換： 常用函數如 df.dropna() 刪除缺失值、df.fillna(val) 填補空缺值、df.apply(func) 對列或行套用函數、df.groupby("col").agg(func) 進行分組聚合等。常見錯誤： 索引鍵錯誤 (KeyError) 由於欄位名稱拼寫不對；數據類型不匹配導致運算錯誤（如將字串當數字計算）。調試時可用 df.dtypes 檢查各欄位型別，必要時用 pd.to_numeric 或 astype() 轉換。
資料合併與重塑： Pandas 支援 merge 進行資料表合併，類似 SQL 的 JOIN：pd.merge(df1, df2, on="key") 將兩表按鍵合併。使用 df.concat([df1, df2]) 在列方向或欄方向拼接資料。重塑 (Reshape) 操作用於調整資料表結構，例如 df.melt() 將寬格式轉換為長格式，df.pivot_table() 創建樞紐分析表。這些操作有助於整理數據以進行分析或建模。常見錯誤： 合併時鍵不唯一或類型不匹配導致不正確結果，或忽略 how 參數預設 INNER JOIN 導致資料遺失。務必檢查合併結果的行數和預期是否一致。
特徵預處理 (Feature Preprocessing)： 在建模前通常需要對數據進行清洗與特徵轉換：
標準化/正規化 (Normalization): 將數值特徵縮放到相似範圍，提高模型收斂。例如將像素值從 [0,255] 線性縮放到 [0,1]
file-cuatb2ef6rmab8yk5hyvgu
。
標準化 (Standardization): 減去平均值除以標準差，使特徵呈現標準常態分布。Scikit-learn 提供 StandardScaler 實現。
類別編碼: 將分類型資料轉為數值。如 One-Hot 編碼 將類別轉為二進位向量 (使用 Pandas pd.get_dummies() 或 sklearn 的 OneHotEncoder)。注意避免 虛擬變量陷阱（對線性模型可去掉一列避免共線性）。
缺失值處理: 針對空缺數據，可選擇刪除含缺失值的行列，或用平均值/中位數/眾數填補，或更先進的方法如插值。Pandas 的 fillna 與 sklearn 的 SimpleImputer 可輔助處理。
常見錯誤與提示： 須確保訓練集和測試集使用相同的縮放/編碼方式（例如先對整個訓練集計算縮放參數，再應用於測試集）。可以使用 Pipeline 組合預處理和模型，避免遺漏步驟。若看到模型在測試時表現異常，檢查是否忘記應用相同的預處理轉換。
資料集拆分 (Train/Test Split)： 將數據按比例拆分為訓練集與測試集（有時還會再劃分出驗證集）。常用 sklearn.model_selection.train_test_split，確保模型評估的公正性。範例：
python
複製
編輯
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
上例將數據按8:2拆分並設定隨機種子確保可重現。對分類問題可使用 stratify=y 參數保持類別比例平衡。最佳實踐： 切分前先隨機打亂數據，以免原始數據有順序性。同時注意不可泄露測試資訊到訓練過程（如不可用整個資料計算均值再標準化訓練和測試，應只用訓練集資訊）。
數據可視化 (Data Visualization with Matplotlib & Seaborn)
Matplotlib 基本繪圖： Matplotlib 是 Python 最基礎的繪圖庫。常用 pyplot 介面：
python
複製
編輯
import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,4,9,16], label="y=x^2")  # 繪製折線圖
plt.title("Sample Plot")                       # 標題
plt.xlabel("x") 
plt.ylabel("y")
plt.legend()                                   # 顯示圖例
plt.show()
圖表類型： 折線圖 (plt.plot)、散點圖 (plt.scatter)、柱狀圖 (plt.bar)、直方圖 (plt.hist)、盒鬚圖 (plt.boxplot) 等，用於呈現不同數據關係。可以透過 plt.subplot 或 plt.subplots 創建子圖，以在一張圖中呈現多個子圖。常見錯誤： 忘記呼叫 plt.show()（在某些環境需明確顯示）、試圖繪製空數據導致無輸出。調整坐標軸或標籤時可使用 plt.xlim, plt.ylim 或 plt.xticks, plt.yticks 控制範圍與刻度。
Seaborn 高級繪圖： Seaborn 基於 Matplotlib，提供更簡潔美觀的統計圖形。典型用法：
python
複製
編輯
import seaborn as sns
sns.histplot(data=df, x="age", hue="gender", kde=True)  # 帶核密度曲線的直方圖
sns.boxplot(data=df, x="category", y="value")           # 分組盒鬚圖
plt.show()
特點： Seaborn 能自動處理 Pandas DataFrame，並整合統計元素如迴歸線、分類色彩等。常用圖包括：heatmap 繪製熱力圖（例如相關係數矩陣）、pairplot 一鍵繪製成對關係、violinplot 類似盒圖但顯示分布形狀等。最佳實踐： 善用 hue 區分類別、col/row 參數繪製分面圖比較多組資料。Seaborn 默認主題美觀，可以透過 sns.set_theme(style="darkgrid") 等切換風格。
圖像保存與調整： 使用 plt.savefig("figure.png", dpi=300) 可將圖以指定解析度保存。調整圖像大小可在繪圖前使用 plt.figure(figsize=(w, h)) 設定。為提高在報告或比賽輸出的清晰度，可增加 DPI 或使用向量格式 (.svg, .pdf) 輸出。常見錯誤： 保存圖像時檔案路徑錯誤導致未生成文件；調整大小或添加子圖時對象用錯 (區分使用 plt.figure 返回的 Figure 對象與 Axes 對象的操作)。
範例：混淆矩陣熱力圖： 混淆矩陣是評估分類模型的重要視覺化工具。我們可以利用 Seaborn 的 heatmap 來繪製。假設有混淆矩陣：
python
複製
編輯
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cm = np.array([[50, 2], [5, 43]])  # 二分類混淆矩陣
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0","Pred 1"], 
            yticklabels=["True 0","True 1"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
上例生成一個 $2\times2$ 的熱力圖，annot=True 顯示數字，使用藍色調色盤方便識別高低。透過這種方式，可以直觀了解分類錯誤類型：對角線為正確數量，非對角線為錯誤數量。
機器學習基礎與 Scikit-learn 應用 (Classical ML with Scikit-learn)
監督式學習 (Supervised Learning)： 給定帶標籤的訓練數據，學習輸入到輸出之間的映射關係。
線性迴歸 (Linear Regression)： 用於迴歸任務，假設輸出與輸入特徵呈線性關係。模型形式：$\hat{y} = w^T x + b$，通過最小化均方誤差 (MSE) 來找到最佳參數。
mcs.mo
Scikit-learn 用法：
python
複製
編輯
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)                      # 訓練線性迴歸模型
print(model.coef_, model.intercept_)             # 輸出權重和截距
y_pred = model.predict(X_test)
關鍵點： 適用於特徵與目標近似線性關係的情況。常見錯誤： 特徵未標準化可能導致不同尺度下係數難以比較；存在明顯異常值時均方誤差敏感，可以考慮使用均絕對誤差 (MAE) 作為優化目標或模型訓練前先處理離群值。
羅吉斯迴歸 (Logistic Regression)： 用於二元或多元分類，通過 sigmoid 或 softmax 函數將線性組合轉換為概率。二分類中模型為：$\hat{y} = \sigma(w^T x + b)$，$\sigma$ 為 Sigmoid 函數。對數幾率迴歸本質是線性分類模型，使用交叉熵損失訓練。在 sklearn 中：
python
複製
編輯
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)           # 訓練羅吉斯迴歸分類器
prob = clf.predict_proba(X_test)    # 輸出概率
pred = clf.predict(X_test)          # 輸出預測標籤
應用場景： 特徵和類別呈線性可分時效果好，可解釋性強（權重可看作特徵對預測影響）。常見陷阱： 資料高度線性可分時，迴歸係數可能發散（可加 正則化 解決，sklearn 默認有 L2 正則化）。對於多分類，sklearn 默認採用 one-vs-rest 策略，也可設定 multi_class='multinomial' 使用 softmax。
K 最近鄰 (K-Nearest Neighbors, KNN)： 一種非參數的分類/迴歸方法。分類時，對新樣本，在訓練集中找到距離最近的 K 個鄰居，用鄰居中最多的類別作為預測；迴歸時用鄰居均值。sklearn 用法：
python
複製
編輯
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
特點： 簡單直觀，對決策邊界複雜的問題有效；缺點是計算開銷大、無法給出特徵重要性且對高維資料效能下降（維度詛咒）。調參建議： 選擇適當的 K 值避免過擬合/欠擬合，一般通過驗證集或交叉驗證選取。可使用不同距離度量（歐式、曼哈頓等）根據資料特性調整。
決策樹 (Decision Tree)： 以樹狀結構進行決策的模型，可用於分類和迴歸。通過遞迴二分資料集，使每次劃分最大程度降低不純度（分類常用 資訊增益/基尼係數，迴歸用方差降低）。範例：
python
複製
編輯
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
優點： 可解釋性強（可視化決策樹節點決策規則）、能處理非線性關係。缺點： 容易過擬合（樹可能長得過深，需透過 max_depth, min_samples_split 等參數剪枝控制）。常見錯誤： 資料若存在 類別不平衡，純用資訊增益劃分可能導致偏向多數類，需適當調整樣本權重或評估指標。
隨機森林 (Random Forest)： 集成學習方法，透過構建多顆訓練集子集上的決策樹並投票（分類）或平均（迴歸）提升預測穩健性
mcs.mo
。相較單顆樹，隨機森林能減少過擬合並處理高維特徵。sklearn 用法：
python
複製
編輯
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_  # 每個特徵的重要度
優點： 通常有不錯的預測表現，對缺失值和異常值不敏感，並可評估特徵重要性。注意： 過多的樹會增加計算時間，但一般不會降低效果；可適當調整 n_estimators 平衡效能與速度。
支援向量機 (SVM)： 強大的分類（亦可用於迴歸 SVR）模型，通過在高維特徵空間尋找最大間隔的決策超平面分隔資料
ioai-official.org
。可使用核函數 (kernel) 處理非線性分類（如 RBF 核對應高斯函數映射）。sklearn 用法：
python
複製
編輯
from sklearn.svm import SVC
svc = SVC(kernel='rbf', C=1.0, gamma='scale')
svc.fit(X_train, y_train)
特點： 在中小型資料集上效果好，特別是決策邊界複雜但維度不高的情況。缺點： 訓練時間隨樣本和特徵數呈多項式增長，不適合非常大規模資料；參數 C（懲罰項係數）與核參數需要調整。調試提示： 標準化特徵通常對 SVM 很重要；若出現過擬合可降低 C 值，欠擬合則提高 C 或調整核函數參數。
非監督式學習 (Unsupervised Learning)： 資料沒有標籤，模型試圖發現資料內在結構。
K-Means 聚類： 將樣本劃分成 K 個群集，使群集內部相似度高、群集之間差異大。演算法透過隨機初始化 K 個聚類中心，反覆指派樣本到最近中心、更新中心位置直至收斂。sklearn 用法：
python
複製
編輯
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)   # 對每個樣本給出聚類標籤
centers = kmeans.cluster_centers_
注意： K-Means 對初始中心和K值較敏感，常用多次初始化 (n_init) 取最佳結果。輸出 labels 可用於觀察聚類結果，或與真實標籤比較計算 純度 等指標。常見錯誤： 資料若有離群點或非球狀分布，K-Means 聚類效果不佳；可考慮使用 DBSCAN（基於密度的聚類）或 階層式聚類 等替代方法。
主成分分析 (PCA)： 一種降維技術，通過線性變換將原始特徵空間投影到較低維的正交主成分空間，最大限度保留數據變異
ioai-official.org
。前幾個主成分承載了數據的大部分方差。sklearn 用法：
python
複製
編輯
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)   # 將特徵降到2維
print(pca.explained_variance_ratio_)  # 解釋的方差比例
應用： 可用於資料可視化（將高維資料投影到2D/3D繪圖）、降低模型計算負擔或緩解多重共線性。注意： PCA是無監督的，只關注重建誤差不考慮標籤資訊，在降維後需檢查是否保留足夠對區分有用的訊息。延伸： t-SNE 是另一種非線性降維方法，適合可視化高維資料在低維度的分佈，但計算較慢且只用於可視化不宜用於後續模型。
模型評估與調試 (Model Evaluation & Tuning)
評估指標 (Evaluation Metrics)： 根據問題性質選擇合適的評估指標：
準確率 (Accuracy)： 分類中預測正確的比例，公式：$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{總樣本數}}$。適用於類別分布均衡且關心整體正確率的情況。
精確率 (Precision)： 在預測為正的樣本中實際為正的比例，$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$。當誤報代價高（需要控制假陽性）時重視此指標。
召回率 (Recall)： 在所有實際為正的樣本中被正確預測的比例，$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$。當漏檢代價高（關注假陰性）時很重要。
F1 分數： 精確率與召回率的調和平均 $F1 = 2 \cdot \frac{P \times R}{P + R}$，綜合考量精確率和召回率的平衡，適用於類別不平衡或希望兼顧兩者的情況。
均方誤差 (MSE) / 均絕對誤差 (MAE)： 迴歸問題常用指標，分別計算預測值與真實值差的平方均值和絕對值均值。MSE 對離群值懲罰更大，MAE 更健壯。
在 sklearn.metrics 模組中提供了上述指標的實現，如 accuracy_score, precision_score, recall_score, f1_score, mean_squared_error 等，可直接使用。另外，也可使用 classification_report 一次性輸出主要分類指標。範例：
python
複製
編輯
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
注意： 若類別不平衡，準確率可能會掩蓋問題（例如99%負類1%正類，全部預測負有99%準確但召回率低）。此時應輔以精確率、召回率或ROC曲線和AUC等指標全面評估。ROC曲線描繪分類器在各種閾值下真陽率對假陽率的曲線，其下方積分即 AUC (Area Under Curve) 值，AUC 越接近1表示模型區分能力越強。
混淆矩陣 (Confusion Matrix)： 是分類結果的全面呈現。以二分類為例矩陣為 2x2，列通常表示預測類別，行表示真實類別。元素含義：真陽性 TP（實際正且預測正）、假陽性 FP（實際負但預測正）、假陰性 FN（實際正但預測負）、真陰性 TN（實際負且預測負）。通過混淆矩陣可衍生出精確率、召回率、特異度等指標並分析模型在哪種類別易出錯。例如上節所示的熱力圖視覺化有助於分析錯誤類型。
過擬合與欠擬合 (Overfitting vs Underfitting)：
過擬合是指模型在訓練集上表現極佳但無法泛化到未見資料，在其他資料上表現較差的現象
zh.wikipedia.org
。過擬合通常發生於模型複雜度過高、參數過多相較於訓練資料量不足時
zh.wikipedia.org
。此時模型方差大、對訓練資料噪音過度適應。相反，欠擬合是模型過於簡單，以致無法在訓練集上有效學習資料結構的現象
zh.wikipedia.org
；此時模型偏差大。例如，使用線性模型擬合非線性關係會欠擬合。判斷方法： 通過學習曲線觀察訓練和驗證誤差隨樣本量變化；若訓練誤差遠低於驗證誤差且驗證誤差很高，是過擬合；若兩者都很高，是欠擬合。 防止過擬合的實踐： 使用更多訓練數據、特徵選擇/降維減少不相關特徵、對模型引入正則化（如 L1/L2 項限制權重大小）、使用交叉驗證調參防止過度依賴單一驗證集結果、採用早停 (Early Stopping) 在驗證集性能開始下降時停止訓練等措施
ioai-official.org
ioai-official.org
。對深度學習模型，可使用 Dropout 隨機丟棄部分神經元以正則化
ioai-official.org
，或採用數據增強增加資料多樣性。
超參數調整 (Hyperparameter Tuning)： 機器學習模型有許多需事先設定的參數（如樹的深度、正則化係數、學習率等），稱為超參數。調參是提升模型性能關鍵步驟。常用方法：
Grid Search (網格搜尋)： 枚舉所有組合的超參數值進行嘗試，例如 sklearn.model_selection.GridSearchCV 可對指定參數網格進行交叉驗證評估，選出最佳參數組合。
Random Search： 在給定範圍內隨機取部分組合嘗試，比網格更高效地探索大範圍參數空間。
貝葉斯優化 等高級方法利用歷次評估結果漸進選擇下一組測試的參數，提高搜索效率。
實踐建議： 先調整對結果較敏感的參數，如樹模型的樹數和深度、神經網絡的學習率等。使用 交叉驗證 (Cross-Validation) 評估參數組合的泛化性能
mcs.mo
。交叉驗證將訓練資料多折劃分，多次訓練評估取平均，使得調參結果更穩健。當資料量大時，可使用較少折數（例如 5-fold CV）平衡計算負荷。
交叉驗證 (Cross-Validation)： 如上述，為了更可靠地評估模型泛化能力，將訓練集拆分成 $k$ 個子集，其中 $k-1$ 個用於訓練，剩下1個用於驗證，重複 $k$ 次使每個子集都做過驗證集，最後對結果取平均
mcs.mo
。優點： 充分利用數據且評估結果更穩定。常用: sklearn.model_selection.KFold 或 StratifiedKFold (分層抽樣用於分類保持類別比例)，或者更高層級的 cross_val_score 直接對給定模型和資料返回交叉驗證分數。注意： 交叉驗證僅用於評估/選參數，最終模型可在使用最佳參數下重新在全部訓練集上訓練。
深度學習與 PyTorch 基礎 (Deep Learning & PyTorch Basics)
感知器與神經網絡基礎： 感知器 (Perceptron) 是最簡單的神經網路，進行線性分類。多個感知器層疊形成多層感知器 (MLP)，屬於前饋神經網絡。每層由若干神經元 (Neuron) 組成，神經元接收前一層輸出的加權和並經過激活函數產生輸出。深度學習通過多層非線性變換，自動從資料中提取高級特徵。
ioai-official.org
ioai-official.org
梯度下降與反向傳播： 梯度下降 (Gradient Descent) 是優化神經網絡參數的基礎演算法。透過計算損失函數相對每個參數的偏導數（梯度），沿著梯度的反方向更新參數以減小損失
ioai-official.org
。反向傳播 (Backpropagation) 高效地計算多層網絡的梯度：從輸出層誤差開始，依照鏈式法則將誤差梯度向後傳遞至各層，獲取每層權重的梯度
ioai-official.org
。PyTorch 和 TensorFlow 等框架自動實現了反向傳播 (autograd)，使用者只需定義前向計算和損失。
激活函數 (Activation Function)： 激活函數引入非線性，使神經網絡可以逼近非線性關係
ioai-official.org
。常見激活函數：
ReLU (線性整流)： $f(x) = \max(0, x)$，簡單高效，對正輸出為線性，負輸出為0。優點是計算梯度不會飽和（正區域梯度=1），收斂快
ioai-official.org
。
Sigmoid (邏輯函數)： $f(x) = \frac{1}{1+e^{-x}}$，將輸出壓縮在 (0,1)，適合二分類輸出概率。但輸入值絕對值較大時梯度趨近0，易出現梯度消失問題。
Tanh (雙曲正切)： $f(x) = \tanh(x)$，輸出在 (-1,1)，類似 Sigmoid 但對稱於0，梯度性質稍好但仍有飽和區問題。
Leaky ReLU / ELU / PReLU： ReLU 的變種，為負區域引入非零斜率（如 Leaky ReLU: f(x)=x*0.01 if x<0），緩解 ReLU 梯度為零的問題。
選擇建議： 隱層常用 ReLU 及其變種；輸出層若為分類，二分類用 Sigmoid 配合二元交叉熵損失，多分類用 Softmax（在 CrossEntropyLoss 中隱含計算）；迴歸問題的輸出層則通常用線性激活（不施加非線性）。
損失函數 (Loss Function)： 衡量模型預測與真實目標的不一致程度，是訓練時要最小化的目標。
ioai-official.org
 常用損失：
均方誤差 (MSE)： $ \frac{1}{n}\sum (y_{\text{pred}} - y_{\text{true}})^2$，迴歸問題常用。
ioai-official.org
偏好平滑的預測，對離群值敏感。
平均絕對誤差 (MAE) 或 Huber 損失：迴歸中更健壯的選擇，MAE 對每個點線性懲罰誤差，Huber 在誤差小時類似 MSE，大時類似 MAE，兼具優點。
交叉熵損失 (Cross-Entropy)： 分類問題使用，度量預測分布與真實分布之間的差異。二分類的交叉熵即對數損失：$-\frac{1}{n}\sum[y \log p + (1-y)\log(1-p)]$。多分類情況下通常和 softmax 結合使用，PyTorch 的 nn.CrossEntropyLoss 已整合 softmax 計算。
其他： 如負對數概似損失 (NLL)、KL 散度、hinge loss (SVM使用) 等，視具體任務選擇。
優化器 (Optimizer)： 決定如何根據損失函數的梯度更新模型參數。基本的隨機梯度下降 (SGD) 每次用一個或一批樣本計算梯度並更新參數。進階優化演算法在 SGD 基礎上做改進：
Momentum 動量法： 在更新中加入上一次更新的動量，幫助平滑更新方向，加速收斂並減少局部震盪。
Adam： 自適應學習率方法，結合動量和 RMSProp 的思想，為每個參數維持一階與二階矩估計來調整學習率
ioai-official.org
。通常具有較快的收斂速度和較少的參數調整需求，在各種網絡中被廣泛使用。PyTorch torch.optim.Adam 可直接使用。
AdamW： Adam 的改進版本，正則化方式改進，更利於控制模型泛化。
學習率調整： 設定合適的學習率 (learning rate)至關重要：太大導致發散，太小導致收斂慢陷入局部最優。可採用學習率衰減策略，如每若干epoch將學習率乘以一個因子，或使用調度器 (torch.optim.lr_scheduler) 自動調整。
實踐： 深度學習常以 Adam 作為初始優化器。如訓練不穩定再嘗試 SGD 配合學習率調整或其他優化器。訓練過程中監控損失下降趨勢，如長時間停滯可適當降低學習率。
PyTorch 基本操作與語法： PyTorch 是深度學習框架，採用動態計算圖，使用張量 (Tensor) 作為基本數據結構（類似 NumPy 陣列但可在 GPU 加速）。
Tensor 建立與操作： torch.tensor(data) 將 Python 結構轉為張量，或使用 torch.randn, torch.zeros 等直接創建。張量有 shape 屬性，運算介面與 NumPy 類似。可以使用 tensor.view() 或 tensor.reshape() 改變形狀。張量的資料型態很重要：整數型用於標籤，浮點型用於特徵/權重計算等。如需，使用 tensor.long(), tensor.float() 進行型別轉換。
GPU 加速： 使用 device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 判斷是否有 GPU。
file-cuatb2ef6rmab8yk5hyvgu
將模型和資料 .to(device) 可以把計算移到 GPU 上。注意，移到 GPU 的對象需全部在同一設備上運算。調試提示： 如果遇到 tensor 不在同一裝置的錯誤（如 trying to operate on CPU tensor and GPU tensor），檢查確保所有張量和模型都已 .to(device)。
再現性 (Reproducibility)： 為確保結果可重現，可設定隨機種子：例如 torch.manual_seed(42), np.random.seed(42)，並盡量固定其他隨機性來源（如確保每次訓練數據順序固定或使用 torch.backends.cudnn.deterministic = True 等）。不過在 GPU 上完全重現可能需要額外設定以犧牲部分性能。
Dataset 與 DataLoader： PyTorch 提供 torch.utils.data.Dataset 接口來定義數據集，DataLoader 用於批量迭代數據集。常用 TensorDataset 封裝張量資料集，搭配 DataLoader 可輕鬆迭代批次：
python
複製
編輯
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(features_tensor, labels_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch_features, batch_labels in loader:
    # 訓練循環中使用 batch 資料
    ...
DataLoader 幫助自動打亂數據 (shuffle=True) 並將資料切成小批，有助於梯度下降的穩定和效率。常見錯誤： 如果 DataLoader 輸入的是 numpy 陣列而非 torch.Tensor 會報型別錯，需先轉成 Tensor；自定義 Dataset 實現 __len__ 和 __getitem__ 時也需注意索引邊界。
神經網路構建 (nn.Module)： PyTorch 透過 torch.nn.Module 定義模型結構。可繼承 Module 並在 __init__ 方法中定義各層，在 forward 方法中定義前向傳播。範例，構建一個兩層全連接網路進行三分類：
python
複製
編輯
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一層全連接
        self.act1 = nn.ReLU()                       # 激活函數 ReLU
        self.fc2 = nn.Linear(hidden_dim, output_dim) # 第二層全連接 (輸出層)
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.fc2(x)  # 輸出層通常不加激活(交由損失函數處理)
        return x

model = SimpleNN(input_dim=4, hidden_dim=16, output_dim=3).to(device)
print(model)
輸出模型結構示例：
scss
複製
編輯
SimpleNN(
  (fc1): Linear(in_features=4, out_features=16, bias=True)
  (act1): ReLU()
  (fc2): Linear(in_features=16, out_features=3, bias=True)
)
常見錯誤： 在 forward 中操作張量時，如果使用原地操作 (帶 _ 的函數如 x.relu_()) 可能導致梯度跟蹤問題，除非確信必要否則使用非原地版本。還有，Module 屬性需是 PyTorch Layer/Module，否則不會被納入參數（如未將 nn.Linear 賦給 self. 開頭變數，將不會被模型識別）。調試時可用 model.parameters() 確認所有應訓練參數都在其中。
模型訓練迴圈： 典型的訓練步驟包括：前向計算 -> 計算損失 -> 反向傳播 -> 更新參數。每個 epoch（訓練集完整迭代一次）都重複以上步驟，並可在適當時機評估模型表現。以下是通用的 PyTorch 訓練迴圈結構：
python
複製
編輯
criterion = nn.CrossEntropyLoss()                         # 定義損失函數 (例如交叉熵)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 定義優化器 (例如 SGD)
num_epochs = 5

for epoch in range(num_epochs):
    model.train()                             # 設置為訓練模式
    for X_batch, y_batch in train_loader:     # 遍歷每個批次
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()                # 重置梯度
        outputs = model(X_batch)             # 前向傳播計算輸出
        loss = criterion(outputs, y_batch)   # 計算當前批次的損失
        loss.backward()                      # 反向傳播計算梯度
        optimizer.step()                     # 更新參數
    # 可在這裡計算並打印訓練集和驗證集的損失、準確率等指標
    model.eval()                             # 切換為評估模式
    # (驗證模型性能的代碼與上面類似，但不需要 backward 和 step，且要包裹 no_grad)
註解說明： 在 model.train() 模式下，像 Dropout、BatchNorm 層會啟用訓練行為；而 model.eval() 下它們會固定參數或統計值，用於評估。每個 batch 前用 optimizer.zero_grad() 清除前一次的梯度累積。loss.backward() 自動計算所有參數的梯度，optimizer.step() 則依據選定的優化策略調整參數。常見錯誤： 忘記 zero_grad 導致梯度累加過頭；沒有 model.eval() 在驗證時停用dropout導致評估不準確；未 with torch.no_grad(): 包裹驗證計算，可能導致不必要的緩存佔用和速度降低。
訓練過程監控與保存： 在訓練中定期監控損失和評估指標的變化。如果發現驗證集損失開始上升，說明過擬合開始，可考慮早停。使用 torch.save(model.state_dict(), "model.pt") 可以保存模型權重，以便賽後或部署時使用；相應地用 model.load_state_dict() 加載。還可保存 Optimizer state 以便從中斷處繼續訓練。比賽中通常在本地保存模型後再打包提交。
卷積神經網絡與計算機視覺 (CNN & Computer Vision)
卷積層 (Convolutional Layer)： 卷積神經網絡 (CNN) 專門用於處理影像等網格結構數據。卷積層使用卷積核/濾波器 (filter/kernel) 在輸入上滑動，進行點積運算提取局部特徵
hackmd.io
。每個卷積核可視為檢測影像中特定模式的探測器
hackmd.io
。卷積運算本質：滑動 + 內積
hackmd.io
——卷積核在輸入圖像上按步長逐像素移動，在每個位置與重疊的區域逐元素相乘再相加，結果形成特徵圖 (feature map)
hackmd.io
。透過多個不同卷積核，卷積層能同時抽取輸入的多種特徵，如邊緣、紋理等。關鍵參數： 卷積核大小 (如3x3, 5x5)、輸出通道數（卷積核個數）、步幅 (stride) 控制滑動步長、填充 (padding) 在輸入邊界補零以控制輸出尺寸。 PyTorch 實現： 使用 nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) 定義卷積層。例如 nn.Conv2d(1, 16, kernel_size=3, padding=1) 表示輸入通道1（灰度圖），輸出16個特徵圖，卷積核3x3，周圍補零一圈保持尺寸。卷積輸出高度寬度計算公式：$H_{out} = \frac{H_{in} - K + 2P}{S} + 1$（其中 $K$ 是核大小，$P$ 是padding，$S$ 是stride）。常見錯誤： 未對輸入加正確的維度（PyTorch 卷積輸入需是四維張量 [batch, channels, height, width]，單張圖片也要增加 batch 維度）；不同層特徵圖大小不匹配，如卷積後尺寸非預期導致後續全連接層維度錯誤。可透過 print(tensor.shape) 或 model.summary()（如有）排查每層尺寸。
池化層 (Pooling Layer)： 池化通過取局部區域統計值來縮減特徵圖尺寸，常用於降採樣降低計算量並引入位置不變性。主要類型：最大池化 (Max Pooling) 取區域內最大值，平均池化 (Average Pooling) 取平均值。池化層通常使用 $2\times2$ 覆蓋區域配合步幅2，使寬高減半。
hackmd.io
池化的作用包括減少參數、抑制噪聲和對輸入輕微變換（如平移、旋轉、縮放）的容忍
hackmd.io
。例如，2x2 max pooling 可使圖像尺寸縮小一半，同時保留主要特徵，提升模型對位置變動的特徵不變性
hackmd.io
。 PyTorch 實現： 使用 nn.MaxPool2d(kernel_size, stride)，常見例如 nn.MaxPool2d(2, 2)。平均池化則是 nn.AvgPool2d。注意： 卷積層和池化層都會影響感受野 (Receptive Field)，即網絡輸出節點能看到的輸入區域範圍，多層卷積/池化可擴大感受野讓模型學到更抽象的特徵
hackmd.io
。常見錯誤： 池化窗口過大導致過度壓縮資訊；池化一般無參數，但要注意是否會導致維度非整數（通常選步幅等於窗口大小避免重疊或遺漏）。
CNN 經典架構： CNN 通常由卷積層 + 激活 + 池化組成模塊，堆疊若干次後接全連接層進行分類
hackmd.io
hackmd.io
。早期經典網絡如 LeNet-5 用於手寫字識別，包含兩組卷積+平均池化，最後兩層全連接分類；AlexNet, VGG 加深了卷積層並使用 ReLU 和 MaxPool，ResNet 引入殘差連接。這些架構透過增加深度或寬度提升影像識別效果。實踐： 小型任務可參考 LeNet 結構：例如輸入28x28灰度圖，可用[Conv(1->6)+ReLU+Pool] -> [Conv(6->16)+ReLU+Pool] -> 展平 -> [FC(1655->120)+ReLU] -> [FC(120->84)+ReLU] -> [FC(84->10)] 輸出10類。當然，也可調整通道數和層數，均衡模型複雜度與數據集大小。
轉移學習 (Transfer Learning)： 在資料有限的情況下，使用在大規模資料上預訓練好的模型（如 ResNet、MobileNet 等）作為特徵提取器，再微調 (fine-tune) 用於新任務
mcs.mo
。例如，PyTorch 提供 torchvision.models 預訓練模型：
python
複製
編輯
from torchvision import models
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False       # 冻結預訓練權重
model.fc = nn.Linear(512, num_classes) # 取代最後一層為新任務
如上將 ResNet18 最終全連接層改為適應新類別數並只訓練該層（其他層參數凍結保持預訓練值），能在小數據上取得不錯效果。優點： 省去訓練大模型的成本，利用預訓練網絡已學得的通用特徵（如邊緣、形狀），收斂快且精度高。實際比賽中，如果允許，可使用預訓練模型微調來提高圖像任務表現。
圖像資料增強 (Data Augmentation)： 增強技術透過對訓練影像做隨機變換來合成新樣本，減少過擬合
mcs.mo
。常用方法：隨機翻轉（水平/垂直）、隨機裁剪、旋轉、調整亮度/對比度、加雜訊等。PyTorch 可用 torchvision.transforms 定義增強流水線，例如：
python
複製
編輯
import torchvision.transforms as T
transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor()
])
dataset = torchvision.datasets.ImageFolder("data/train", transform=transform)
這將對每張圖片隨機裁剪到224x224、隨機水平翻轉，以及隨機調整亮度對比後轉為張量。提示： 增強只應作用於訓練集，保持驗證/測試集不變以正確評估模型性能。
前沿模型簡述： (比賽理論範圍涉及，僅需理解概念)
YOLO (You Only Look Once) / SSD (Single Shot Detector)： 主流**目標檢測 (Object Detection)**模型，直接從影像一次性回歸出多個邊界框及其類別
ioai-official.org
。YOLO 將整張圖像分網格，每格預測框和類別；SSD 使用多尺度特徵圖預測。與分類不同，檢測模型輸出目標的位置和類別。應用： 自動駕駛中的行人車輛檢測等。
U-Net： 一種**圖像分割 (Image Segmentation)**模型，典型的編碼器-解碼器架構，中間使用跳躍連接結合高低層特徵，能對每個像素做分類（區分不同物體或區域）
ioai-official.org
。常用於醫學影像分割等任務，需要輸出與輸入同尺寸的像素標註。
GAN (生成對抗網絡)： 包含生成器 (Generator) 和判別器 (Discriminator) 兩個網絡的框架。生成器試圖產生以假亂真的圖像，判別器試圖分辨真實圖像與生成圖像，雙方通過對抗訓練共同進步。GAN 能生成極為逼真的圖像，是圖像生成、風格轉換等應用的核心方法。
自我監督學習 (Self-Supervised Learning)： 不依賴人工標籤，通過設計預測輸入一部分內容的任務來學習影像特徵。例如圖像拼圖重排、遮擋區域恢復等任務。這些技術能利用海量未標記圖片進行預訓練，再遷移到下游任務。
視覺Transformer (ViT)：將 Transformer 架構應用於影像，把圖像切成塊（如 16x16 pixel patch），每塊當作詞嵌入，通過自注意力機制建模全局關係。ViT 在大數據集上可以達到與 CNN 相當甚至更好的效果，是近年視覺領域的研究熱點
ioai-official.org
。
CLIP： OpenAI 提出的跨模態模型，將圖片和文字映射到共同的向量空間。透過對大量圖文配對資料的對比學習，CLIP 能理解圖像內容和文字描述間的對應關係，可用於以文字搜尋圖片等任務。
Stable Diffusion / DALL·E： 近年流行的生成式模型，能根據文字輸入產生高品質圖像。DALL·E 使用 Transformer 模型直接學習圖文關係；Stable Diffusion 則結合擴散模型和潛在空間的概念，能在較快速度下生成圖像。這類模型參數龐大，屬於預訓練後使用的範疇，在理論上屬生成模型部分，比賽可能涉及概念理解。
自然語言處理 (NLP) 基礎
文本前處理 (Text Preprocessing)： NLP 任務通常需要先將文字轉為適合模型處理的形式。斷詞/標記化 (Tokenization) 是將句子切分為詞彙或子字串（對英文通常以空格和標點分詞；對中文可用結巴等分詞工具）。可以移除停用詞（常見無意義詞，如“的”、“and”）、轉小寫、去除標點符號等清洗操作。對於詞彙，還有 詞幹提取 (Stemming) 和 詞形還原 (Lemmatization)，將詞彙化簡到共同形態（例如動詞還原為原形），以減少特徵維度。
詞向量與嵌入 (Word Embeddings)： 將文字轉換為數值向量是 NLP 的核心步驟。最簡單的是獨熱編碼 (One-Hot)，每個詞一個維度，高維且無法表達詞語間關係。詞嵌入是一種密集低維表示，使相似詞在向量空間中接近。Word2Vec 與 GloVe 是經典的詞嵌入模型，它們通過在大語料庫上預訓練，使詞向量能捕捉語義關聯（如“king”-“man”+“woman”≈“queen”）。在 PyTorch，可使用 torch.nn.Embedding(num_embeddings, embedding_dim) 層來將詞ID映射到對應的向量；也可載入預訓練詞向量作為初始權重。優點： 嵌入向量顯著提升模型對文本語意的理解能力，比獨熱表示需要更少維度就能表達豐富資訊。
特徵表示與向量化： 除了詞嵌入，傳統方法還包括 n-gram 特徵 和 TF-IDF 向量。Bag-of-Words (詞袋模型) 將文本表示為詞頻向量，不考慮詞序。TF-IDF (詞頻-逆文件頻率) 為詞袋加權，降低常見詞影響，突出關鍵詞。Scikit-learn 用法：
python
複製
編輯
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ["我 喜歡 機器 學習", "機器 學習 很 有趣"]
vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")  # 定義分詞規則
X = vec.fit_transform(corpus)
print(vec.get_feature_names_out())  # 詞彙表
print(X.toarray())                  # TF-IDF 矩陣
TF-IDF 向量常搭配傳統機器學習分類器（如 Naive Bayes、邏輯迴歸、SVM）用於文本分類，在資料量不大時效果不錯且速度快。常見錯誤： 忘記在預處理時處理文字編碼導致讀取錯誤，或 token_pattern 導致分詞不如預期。
常用NLP任務模型：
文本分類： 輸入為文本，輸出為類別（例如垃圾郵件識別、情感分析）。可用機器學習算法（將文本向量化後用如羅吉斯迴歸、SVM）或深度學習模型（如簡單的詞嵌入+平均/池化，再接全連接，或 RNN/CNN 模型）來實現。Scikit-learn 管道範例：
python
複製
編輯
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
pipe = make_pipeline(TfidfVectorizer(), LogisticRegression())
pipe.fit(train_texts, train_labels)
preds = pipe.predict(test_texts)
上述使用 TF-IDF 和 LogisticRegression 的流水線完成文本分類。深度學習方法： 可構建簡單 CNN 在詞序上卷積抽取關鍵詞特徵，或 RNN（如 LSTM）順序讀取詞向量捕捉語序信息，再接全連接分類。常見陷阱： 中文處理需要先切詞；長文本可能需要截斷或分段處理避免輸入過長。
序列標注： 如 分詞、命名實體識別 (NER) 等，給序列中每個元素標籤。常用 Bi-LSTM + CRF 或 Transformer 等架構處理。
機器翻譯 / 文字生成： 傳統上使用 編碼器-解碼器 (Encoder-Decoder) 架構的 RNN 或 LSTM，近年則主要依賴 Transformer 架構。
Transformer 與注意力機制 (Attention)： Transformer 是目前 NLP 主流模型架構，基於自注意力 (Self-Attention) 機制而非循環網絡。注意力機制讓模型在編碼序列時可以學習權重來關注輸入中的不同部分
ioai-official.org
。Transformer 將序列同時餵入多頭注意力層，捕捉詞與詞之間的依賴關係，擺脫了序列計算的瓶頸。優點： 支持並行計算（相較 RNN 須序列計算），效果隨模型變大顯著提升。BERT、GPT 等皆是以 Transformer 為基礎。
預訓練語言模型 (Pre-trained NLP Models)： 如 BERT (雙向編碼器表示) 和 GPT (生成式預訓練變換模型) 是 Transformer 架構的預訓練模型，在海量文本上訓練，學習通用的語言表示。BERT 側重於理解（填詞、分類等下游任務效果突出），GPT 側重於文本生成。
ioai-official.org
 在NLP比賽中，往往通過微調 (Fine-tuning) 這些預訓練模型能取得極佳效果。例：Hugging Face 提供易用接口，可快速載入預訓練模型:
python
複製
編輯
from transformers import BertTokenizer, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
之後將文本轉換為模型輸入格式並訓練即可。注意： 由於參數龐大，預訓練模型調優需要相對較高的計算資源和調參技巧，但其預訓練知識往往能極大提升低資源任務的表現。
大型語言模型與應用： GPT-3/GPT-4 等大型語言模型 (LLMs) 擁有數以十億計參數，基於Transformer架構，能執行各種複雜的NLP任務（翻譯、問答、摘要等），甚至不需要專門微調就能通過提示完成任務 (提示學習/零樣本學習)。當前趨勢還包括：
Prompt Engineering (提示設計)： 為 LLM 設計合適的提示或問題以引導其產生預期結果。
Finetuning 大模型 使用 LoRA、適配器 (Adapter) 等技術：這些方法凍結大部分原模型權重，只訓練小量附加參數，以較低資源調適模型至特殊任務，同時保留大模型原有知識。
ChatGPT, Bard 等聊天機器人： 在大型模型上透過對話數據進行強化學習人類反饋 (RLHF) 微調，使其更善於進行多輪對話和遵循指令。原理涉及策略優化與獎勵模擬，但比賽中理解概念即可。
LLM Agents： 結合 LLM 與工具使用能力的系統，讓模型可以調用插件或API執行操作，例如利用 Python 計算、資料庫查詢等（如 LangChain 框架）。這屬更前沿方向，一般競賽中不涉及實作，但瞭解概念有助於擴展視野。
以上NLP知識在比賽中理論部分可能涉及概念題，例如詢問注意力機制的作用、BERT與GPT的差異等；實作部分則可能聚焦較基本的文本分類或分析任務，因此應熟悉文本處理流程及常用模型。比賽允許查詢 HuggingFace 等資源
mcs.mo
，如遇複雜NLP任務，可查相關庫文件作輔助。
範例題解：手寫數字識別 (CNN 實作範例)
題目描述： 本範例來自 MOAI 比賽提供的手寫數字識別編程題。任務是使用 PyTorch 構建卷積神經網絡（CNN）模型來識別手寫數字（0~9）。數據集為經典的 MNIST 手寫字集：訓練集 60000 張 28x28 灰度圖，測試集 10000 張。以下我們逐小題給出詳細解答，包括數據讀取與預處理、模型構建、模型訓練與評估、以及測試預測與提交格式。
小題 1：數據讀取與預處理 (15 分)
任務要求：
讀取提供的訓練影像檔案 train-images.pt 及訓練標籤檔案 train-labels.csv
file-cuatb2ef6rmab8yk5hyvgu
。
將數據按 8:2 比例劃分為訓練集和驗證集
file-cuatb2ef6rmab8yk5hyvgu
。
對影像數據進行歸一化處理並轉換為 PyTorch 張量
file-cuatb2ef6rmab8yk5hyvgu
。
解題思路： MNIST 數據已給定為 .pt (PyTorch tensor) 和 .csv 格式，直接讀取即可。歸一化可將像素值從 [0,255] 縮放到 [0,1] 浮點數，這是常見的預處理，有助於加速模型收斂。讀取後將資料劃分訓練/驗證集，便於在訓練過程中檢驗模型泛化能力。切分可使用 torch.utils.data.random_split 方便地按給定比例拆分 TensorDataset。 程式實現：
python
複製
編輯
import torch
import pandas as pd
from torch.utils.data import TensorDataset, random_split, DataLoader

# 讀取資料集
images_raw = torch.load('train-images.pt')           # 張量檔案 (60000, 28, 28)
labels_raw_df = pd.read_csv('train-labels.csv')      # CSV檔案包含標籤欄位
labels_raw = labels_raw_df['label'].values           # 提取標籤數組

# 歸一化圖像數據並轉為 float 張量
# MNIST 像素範圍 0-255，除以255使之落在0-1區間
images = images_raw.float() / 255.0                  # shape: (60000, 28, 28), dtype: float32
# 增加一個通道維度，因為是灰度圖，CNN 輸入需要 [批次, 通道, 高, 寬]
images = images.unsqueeze(1)                         # 現在 shape: (60000, 1, 28, 28)
labels = torch.tensor(labels_raw, dtype=torch.long)  # 轉換標籤為 long 張量

# 建立張量資料集並按8:2拆分為訓練集和驗證集
dataset = TensorDataset(images, labels)
train_size = int(0.8 * len(dataset))                 # 80%作訓練
val_size = len(dataset) - train_size                 # 20%作驗證
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 建立數據加載器 (DataLoader) 用於迭代批次數據
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=False)

print(f"總樣本數: {len(dataset)}, 訓練樣本: {len(train_dataset)}, 驗證樣本: {len(val_dataset)}")
# 輸出示例: 總樣本數: 60000, 訓練樣本: 48000, 驗證樣本: 12000
程式講解：
使用 torch.load 直接加載 .pt 檔案中的 tensor。train-images.pt 讀入後大小為 (60000,28,28)，類型可能是 ByteTensor（0~255）。我們用 .float() 轉為浮點型，再 /255.0 進行歸一化。
手寫數字為灰度圖，缺少顏色通道維度，因此用 unsqueeze(1) 在維度1處插入通道=1，得到形狀 (60000,1,28,28)。這對於 nn.Conv2d 等層是必要的。
標籤讀入使用 pandas 主要是因為 CSV 格式方便。讀取後提取 label 欄位轉為 numpy 陣列，再用 torch.tensor 轉為 LongTensor。分類問題中標籤需為整數類別代號，PyTorch 的 CrossEntropyLoss 需要 Long 型。
利用 random_split 按比例拆分資料集，其中訓練集 48000 筆、驗證集 12000 筆（5 萬 + 1 萬 = 60,000）。DataLoader 將資料集包裝成可迭代對象，設定批次大小 128，訓練集 shuffle=True 打亂確保每epoch數據順序隨機，驗證集可不打亂。
檢查點： 此時可從 train_loader 取一個批次檢查資料形狀是否正確，比如 images_batch, labels_batch = next(iter(train_loader))，應得到 images_batch.shape == (128,1,28,28)，labels_batch.shape == (128,)。
小題 2：構建 CNN 模型 (15 分)
任務要求：
定義一個簡單的卷積神經網絡，需滿足：至少兩個卷積層、兩個池化層，使用激活函數，輸出層為10維（對應0~9類別），並定義前向傳播將層連接
file-cuatb2ef6rmab8yk5hyvgu
file-cuatb2ef6rmab8yk5hyvgu
。 解題思路： 根據要求，我們可以設計一個兩層卷積的 CNN。典型結構如下：卷積1 -> 激活 -> 池化 -> 卷積2 -> 激活 -> 池化 -> 展平 -> 全連接輸出層。激活函數選用 ReLU。每個卷積層後緊跟池化層，一方面提取特徵一方面減少特徵圖尺寸。最後全連接層輸出大小為10。下面按照這一思路實現模型。 程式實現：
python
複製
編輯
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定義卷積層和池化層
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        # 第一個卷積: 輸入通道1(灰度)，輸出通道16，5x5卷積核，padding=2確保輸出28x28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 最大池化: 2x2區域取最大值，尺寸減半
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        # 第二個卷積: 輸入通道16，輸出通道32，5x5卷積核，padding=2確保輸出14x14在池化前
        # (經過第一次池化28->14，再conv2後因padding輸出14x14)
        self.fc = nn.Linear(in_features=32 * 7 * 7, out_features=10)
        # 全連接層: 32個特徵圖 * 7*7空間大小 展平後作為輸入, 輸出10個類別
        
        self.activation = nn.ReLU()  # 使用 ReLU 激活函數

    def forward(self, x):
        # 前向傳播定義各層的連接
        x = self.conv1(x)
        x = self.activation(x)   # 卷積後經 ReLU 非線性
        x = self.pool(x)         # 池化層將 28x28 -> 14x14
        x = self.conv2(x)
        x = self.activation(x)   # 第二次 ReLU
        x = self.pool(x)         # 第二次池化 14x14 -> 7x7
        x = x.view(x.size(0), -1)
        # 展平張量: 將(batch, 32, 7, 7)攤平為(batch, 32*7*7)
        x = self.fc(x)           # 輸出層線性變換到10維
        return x

# 初始化模型並移至設備 (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
print(model)
輸出模型結構，驗證包含所需層次：
scss
複製
編輯
CNN(
  (conv1): Conv2d(1, 16, kernel_size=5, padding=2)
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0)
  (conv2): Conv2d(16, 32, kernel_size=5, padding=2)
  (fc): Linear(in_features=1568, out_features=10, bias=True)
  (activation): ReLU()
)
如上所示，兩個卷積 (conv1, conv2)、兩個池化 (pool重複使用)、一個 ReLU 激活函數模組（可重複使用）、和一個輸出全連接層均已定義滿足要求。 模型講解：
卷積層參數選擇： 這裡卷積核取 5x5 且 padding=2，主要是方便計算：輸入28x28經 Conv1 後因補零保持28x28，再 MaxPool2d 2x2->14x14。Conv2 同理，輸入14x14 補零後保持14x14，再池化->7x7。輸出特徵圖數量第一層16、第二層32，是較小網絡設定，當然也可增加。激活函數在每個卷積後使用 ReLU 增加非線性。
池化層作用： 每次池化將特徵圖尺寸減半，最終 28x28 -> 7x7，因此全連接層輸入為3277=1568維。池化也減少計算量和參數，並提供平移縮放不變性，符合題意要求兩個池化層。
全連接層： 將卷積提取的特徵映射到分類輸出。這裡僅用單層線性，亦可視需要加一層隱藏層提升表達力。輸出維度10對應數字0-9類別。注意：如果前面卷積/池化參數改變，展平後輸入維度1568也需相應更改，否則會尺寸不匹配。
常見錯誤檢查： 若 print(model) 或 forward 測試中出現尺寸錯誤，通常是展平維度算錯，建議打印 x.shape 驗證每步尺寸。此模型中 conv1輸出(?,16,28,28) -> pool -> (?,16,14,14) -> conv2 -> (?,32,14,14) -> pool -> (?,32,7,7)，展平即3277=1568對應 fc 輸入。這與代碼一致，如有不符說明卷積參數計算有誤。
小題 3：訓練模型 (20 分)
任務要求：
選擇適當的損失函數（如 MSELoss, CrossEntropyLoss 等）
file-cuatb2ef6rmab8yk5hyvgu
。本問題為多類分類，應選交叉熵損失以衡量預測機率分布與真實分布差異。
選擇優化器（如 SGD, Adam）並設定學習率
file-cuatb2ef6rmab8yk5hyvgu
。Adam 往往收斂較快，SGD 配合適當學習率也可。這裡我們選用 Adam 優化。
將模型在訓練集上訓練至少 5 個 epoch，每個 epoch 結束時分別打印訓練集和驗證集的損失和準確率
file-cuatb2ef6rmab8yk5hyvgu
。
解題思路： 先將模型和數據加載到合適裝置 (GPU 若可用)。定義 criterion 為交叉熵損失（因為我們的輸出未經 softmax，但 PyTorch 的 CrossEntropyLoss 內部會先對輸出做LogSoftmax，所以直接使用即可）。定義 optimizer 為 Adam，設置學習率如 0.001。接著編寫訓練循環：外層跑5個 epoch，內層遍歷 train_loader 做梯度更新。每個 epoch 結束後，用 val_loader 計算驗證損失和準確率。打印結果時，為清晰起見，同時列出當輪的 train/val 指標。 程式實現：
python
複製
編輯
import torch.optim as optim

criterion = nn.CrossEntropyLoss()               # 選擇交叉熵損失函數 (適用於分類)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 選擇 Adam 優化器, 學習率 0.001

num_epochs = 5
for epoch in range(num_epochs):
    model.train()                               # 模型設為訓練模式
    running_loss = 0.0
    correct = 0
    total = 0
    # 訓練循環
    for batch_idx, (images_batch, labels_batch) in enumerate(train_loader):
        # 將當前批次數據移到計算設備
        images_batch = images_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        optimizer.zero_grad()                   # 清空上一批次的梯度
        outputs = model(images_batch)           # 前向傳播得到輸出 (logits)
        loss = criterion(outputs, labels_batch) # 計算當前批次的交叉熵損失
        loss.backward()                         # 反向傳播，計算梯度
        optimizer.step()                        # 更新模型參數
        
        running_loss += loss.item() * labels_batch.size(0)   # 累加損失總和（乘批次大小）
        # 計算準確個數: 預測值為輸出中最大logit的類別
        _, predicted = outputs.max(1)                        # predicted shape: (batch_size,)
        correct += (predicted == labels_batch).sum().item()  # 計算匹配的數量
        total += labels_batch.size(0)
        
        # 可選：每隔若干批次打印一次中間狀態
        if batch_idx % 50 == 0:
            batch_acc = (predicted == labels_batch).float().mean().item()
            print(f"[Epoch {epoch+1} Batch {batch_idx}] 目前批次訓練Loss: {loss.item():.4f}, Accuracy: {batch_acc*100:.2f}%")
    
    # 計算該 epoch 的平均訓練損失和準確率
    train_loss = running_loss / total
    train_accuracy = correct / total
    
    # 驗證循環（不需梯度更新）
    model.eval()                                # 模型設為評估模式
    val_loss_sum = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():                       # 關閉梯度計算
        for images_batch, labels_batch in val_loader:
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)
            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)
            val_loss_sum += loss.item() * labels_batch.size(0)
            _, predicted = outputs.max(1)
            correct_val += (predicted == labels_batch).sum().item()
            total_val += labels_batch.size(0)
    val_loss = val_loss_sum / total_val
    val_accuracy = correct_val / total_val
    
    # 列印本 epoch 的訓練和驗證統計
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy*100:.2f}%, "
          f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy*100:.2f}%")
輸出示例 (模擬數據結果)：
java
複製
編輯
[Epoch 1 Batch 0] 目前批次訓練Loss: 2.3065, Accuracy: 10.16%
[Epoch 1 Batch 50] 目前批次訓練Loss: 0.4458, Accuracy: 85.94%
Epoch 1: Train Loss = 0.5703, Train Acc = 82.45%, Val Loss = 0.1578, Val Acc = 95.20%
Epoch 2: Train Loss = 0.1504, Train Acc = 95.35%, Val Loss = 0.1082, Val Acc = 96.65%
...
Epoch 5: Train Loss = 0.0498, Train Acc = 98.62%, Val Loss = 0.0475, Val Acc = 98.35%
(註：以上數據僅供示意，實際結果視模型隨機初始化和訓練過程而定，但一般CNN在MNIST上可很快達到高於98%的準確率。) 訓練過程講解：
損失函數選擇理由： 多分類問題選擇交叉熵損失 (nn.CrossEntropyLoss)。
file-cuatb2ef6rmab8yk5hyvgu
題目中列出的 MSELoss 也可勉強用於分類（將標籤 one-hot 後當迴歸對待），但那樣效果較差且不符合常規。所以我們正確選用了 CrossEntropyLoss，它內部會對模型輸出做 softmax 後再計算交叉熵，適合此處未經 softmax 的 logits。
優化器選擇與學習率： Adam 是一種自適應學習率的方法，對初學者友好，通常能更快降低loss。我們用 optim.Adam 並設置 lr=0.001（一個常見初始值）。也可以選 SGD，如 optim.SGD(model.parameters(), lr=0.01, momentum=0.9)，但需要適當調整學習率、加上 momentum 等，訓練初期可能略慢。由於題目給了多種選項，選任一並說明理由即可得分。這裡選 Adam 能穩定快速下降 loss。
訓練細節：
我們遍歷每個 batch，累積計算總損失和正確數以便後續計算整個 epoch 平均損失和準確率。running_loss 累積時乘以 batch 大小是為了得到總損失，再除以總樣本算平均。也可以每 batch 平均後累加再平均，結果一致。
predicted = outputs.max(1)[1] 獲得每樣本預測的類別（logits最大索引）。比較 predicted 和真實 labels_batch 得到布林張量，再用 .sum().item()算出 True 數量累加。最終 train_accuracy = correct/total 得到訓練準確率。
列印： 我們按題意在每個 epoch 結束時打印了 Train 和 Val 的 Loss 和 Accuracy
file-cuatb2ef6rmab8yk5hyvgu
。為了監控訓練進展，還在每50個 batch 打印一次當前批次的loss和acc（此非題意要求，但實踐中常見，可幫助了解訓練是否正常進行）。
使用 model.eval() 及 torch.no_grad() 進行驗證，確保不會計算梯度也不會更新模型。同時停用 dropout 等（本模型無 dropout，但習慣如此）。
常見錯誤處理：
忘記 optimizer.zero_grad()： 將導致梯度累積，使得後續更新量不正確。我們每個 batch 都reset梯度避免此問題。
未切換 model.train()/eval()： 若模型有 BN/Dropout層，不切換模式會影響評估結果。我們雖然模型簡單無此類層，仍示範了正確用法。
張量維度不匹配： 若在 loss = criterion(outputs, labels_batch) 報錯，多半是 labels 型別或維度問題。CrossEntropyLoss 要求 outputs shape [batch, classes], labels shape [batch] 且為 Long 型。我們已確保此要求（labels在讀取時轉為Long，輸出經Linear為[batch,10]）。
經過5個epoch訓練，可以看到訓練集和驗證集準確率都逐步上升且接近，表示模型在此資料上效果良好且無明顯過擬合跡象（若訓練Acc遠高於驗證Acc，則可能過擬合）。若需要更高準確率，可考慮增加卷積層通道或訓練更多epoch，但題目要求5個epoch已完成。
小題 4：測試集預測與提交 (50 分)
任務要求：
使用訓練好的模型對測試集進行預測，將結果輸出為 submission.csv 供賽後評分。最終得分依據模型在隱藏測試集的準確率映射到0~50分區間
file-cuatb2ef6rmab8yk5hyvgu
file-cuatb2ef6rmab8yk5hyvgu
。題目給出了準確率對分數的對照，以及產生提交檔案的參考程式
file-cuatb2ef6rmab8yk5hyvgu
file-cuatb2ef6rmab8yk5hyvgu
。 解題思路： 測試影像位於 test_images.pt，格式和訓練影像類似（假設也是 28x28 灰度的 tensor）。步驟：讀取測試影像 -> 進行與訓練相同的歸一化處理 -> 利用已訓練模型預測每張圖片的數字 -> 將預測結果存為 CSV。注意： 要確保對測試的預處理與訓練一致（同樣除255、加通道）。另外，由於測試集沒有標籤，CSV提交格式通常要求兩欄：id 和 label，其中 id 可為樣本的索引。題目程式用 index_label="id" 參數使得CSV第一列為id索引。
file-cuatb2ef6rmab8yk5hyvgu
 程式實現：
python
複製
編輯
# 讀取測試集影像張量
test_images = torch.load('test_images.pt')           # shape: (10000, 28, 28)
# 與訓練集相同的歸一化處理
test_images = test_images.float() / 255.0            # 轉為浮點並歸一化到0-1
test_images = test_images.unsqueeze(1)               # 增加通道維度 -> (10000, 1, 28, 28)

model.eval()                                         # 切換模型為評估模式
with torch.no_grad():                                # 不需要計算梯度
    test_images = test_images.to(device)
    outputs = model(test_images)                     # 對所有測試影像進行前向傳播
    predictions = outputs.argmax(dim=1)              # 取每張圖片最大logit的索引作為預測類別

# 將結果轉為 DataFrame 並保存為 CSV
df_test = pd.DataFrame({"label": predictions.cpu().numpy()})
df_test.to_csv("submission.csv", index_label="id")
print("Test predictions saved, first 5 entries:")
print(df_test.head())
檔案輸出示例： submission.csv 內容格式應該如下：
python-repl
複製
編輯
id,label
0,7
1,2
2,1
3,0
4,4
...
每行表示測試集中對應圖片的預測數字。其中 id 為索引（0開始），label 為模型預測的數字類別。 解答講解：
模型預測過程： 將整個測試集張量一次性傳入模型得到輸出 outputs（形狀 [10000,10]）。使用 argmax(dim=1) 沿類別維度取最大值的索引，即對每個樣本選出概率最高的類別索引作為預測標籤。
file-cuatb2ef6rmab8yk5hyvgu
這裡可以直接全批次預測，是因為模型和顯存能容納10000張28x28灰度圖。若圖像更大或模型更複雜，需像訓練時那樣用 DataLoader 分批預測以免記憶體不足。
CPU/GPU 切換： 由於可能在 GPU 上計算，predictions 仍位於 GPU，要儲存或與CPU上的 pandas互動需要 .cpu() 拷貝回主記憶體，然後再 .numpy() 轉為 numpy 陣列。這點我們在建立 DataFrame 時處理了。
提交格式： 我們創建 DataFrame 時傳入一個 dict {"label": predictions.numpy()} 並指定 index_label="id"，這會自動將 DataFrame 索引作為一列名為 id 輸出。預設索引0~9999正好對應圖片ID，非常方便。注意： 一定要保證提交的順序與測試集原順序一一對應，不要打亂。上述做法直接按原順序輸出，所以是正確的。
額外說明： 題目給出的評分標準
file-cuatb2ef6rmab8yk5hyvgu
顯示：若模型在隱藏測試集準確率 x% 在90-100之間，得分30-50依公式 $50 - 2 * (100 - x)$ 線性遞減；80-90% 對應20-30分；50-80% 對應0-20分；低於50%得0分。因此，本模型若在公開驗證集約98%，在隱藏測試集理論上也接近這水準，那最終得分應在接近滿分50分的區間。
file-cuatb2ef6rmab8yk5hyvgu
 這也是此小題50分的由來。參賽者應盡力優化模型提高準確率以獲得更高得分。 完整流程回顧：
我們完成了從數據讀取、處理，到模型訓練與預測的端到端步驟。在比賽中，可以按照此模板開發模型，並確保在每階段遵循最佳實踐（如正確預處理、調參驗證、結果保存等）。經由實驗，本模型已經能較好地完成手寫數字分類任務。如果時間允許，還可以嘗試：增加訓練 epoch、調整學習率策略、增加 Dropout 正則化防止潛在過擬合、甚至引入卷積層 BN 層等，以期進一步提升效果。但基於題目要求，此解答已足夠達到高評分區間。
結語
本手冊系統彙整了 MOAI 2025 比賽相關的人工智慧知識點，從程式基礎到機器學習理論，從深度學習實作到具體範例解析。在比賽過程中，選手可快速查閱：
語法範例：瞭解 Python、numpy、pandas、PyTorch、sklearn 等庫的常用介面和用法。
理論要點：重溫模型的定義、優缺點、適用場合（如各算法、評估指標、深度學習概念）。
調試提示：遇到錯誤時，對照常見坑點排查修正。
最佳實踐：參考經驗法則（如正則化、資料增強、調參方法）改進方案。
透過範例題的完整解題示範，選手可以掌握如何將上述知識融會貫通地應用於實際問題：包括資料讀取與處理步驟、模型構建與訓練細節、結果輸出與格式要求等。希望本手冊能夠幫助大家在比賽中快速定位知識點並應用自如，取得優異的成績！
