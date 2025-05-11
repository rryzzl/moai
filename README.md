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
