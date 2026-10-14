# Trình bày phần Reinforcement Learning cho thầy

> Xem dạng preview trong VS Code: `Ctrl+Shift+V`, hoặc `Ctrl+K V` để xem song song với file nguồn.

Thầy chuyên về tối ưu và local search nhưng có thể chưa quen RL. Vì vậy cách trình bày dưới đây **dịch RL sang ngôn ngữ tối ưu**: quy hoạch động, hàm đánh giá nước đi, đa dạng hóa và tăng cường.

---

## 0. Ý chính trong một câu

> "Em không thay đổi khung local search. Em chỉ thay **hàm đánh giá nước đi** trong ADD\*: thay vì tỉ lệ $p_i/AW$ chỉ nhìn một bước, em dùng một hàm **cost-to-go học được**. Hàm này ước lượng: *nếu thêm item này, thì về sau incumbent còn cải thiện được bao nhiêu*. Reinforcement learning ở đây thực chất là **quy hoạch động xấp xỉ** cho bài toán chọn nước đi."

---

## 1. Từ điển: RL ↔ ngôn ngữ tối ưu

Nên nói bảng này ngay đầu phần RL. Thầy sẽ thấy hầu hết khái niệm đều đã quen.

| RL | Trong local search / tối ưu | Trong bài của em |
|---|---|---|
| Trạng thái (state) $s$ | Lời giải hiện tại | Vector chọn item $x_S$ + khoảng cách tới incumbent |
| Hành động (action) $a$ | Một nước đi trong láng giềng | Thêm hoặc bỏ một item |
| Chính sách (policy) $\pi$ | Quy tắc chọn nước đi | Chọn nước theo $Q$ |
| Episode | Một lượt chạy local search từ một lời giải xuất phát | 100 nước, xuất phát từ một lời giải GA |
| Phần thưởng (reward) $r$ | Mức cải thiện incumbent sau một nước | $\max(0,\ P(S_{t+1})-P_{best})/\bar p$ |
| Return $G_t$ | Tổng cải thiện incumbent từ bây giờ đến cuối lượt chạy | |
| Hàm giá trị $Q(s,a)$ | **Cost-to-go** của nước đi $a$, tức đánh giá nhìn xa nhiều bước | Điểm số để xếp hạng ứng viên của ADD\* |
| Phương trình Bellman | Đệ quy quy hoạch động | |
| Q-learning | Value iteration dùng **mẫu** thay cho mô hình đầy đủ | |
| DQN | Quy hoạch động xấp xỉ, hàm xấp xỉ là mạng nơ-ron (*neuro-dynamic programming*) | |
| Exploration / exploitation | Đa dạng hóa / tăng cường (diversification / intensification) | $\varepsilon$-greedy, softmax |
| Hệ số chiết khấu $\gamma$ | Trọng số giảm dần cho cải thiện ở xa; tầm nhìn khoảng $1/(1-\gamma)$ bước | $\gamma=0.99$, tức khoảng 100 bước |
| Action masking | Chỉ xét các nước khả thi | Che các nước làm vượt sức chứa |

---

## 2. Kịch bản từng slide

### Mở đầu (chuyển từ slide II.C)

> "Ở phần trước thầy đã thấy: tại mỗi nút ADD\*, ta phải chọn 2 ứng viên để rẽ nhánh, và bài báo chọn theo $p_i/AW$. Đây là một hàm đánh giá **tham lam một bước**. Nó đo lợi ích tức thời của nước đi, nhưng không biết nước đó mở ra hay đóng lại những khả năng nào ở các bước sau. Câu hỏi của em là: có thể **học** một hàm đánh giá nhìn xa hơn không?"

---

### Slide III.A: Reinforcement learning in a nutshell

**Cách trình bày:** đi từ quy hoạch động, vì đó là phần thầy quen nhất.

> "Em xin trình bày RL dưới góc nhìn quy hoạch động.
>
> Việc chọn nước đi trong local search là một **bài toán quyết định tuần tự**. Ở mỗi bước, từ lời giải hiện tại $s$, ta chọn một nước $a$, nhận một mức cải thiện $r$, và chuyển sang lời giải mới $s'$. Về mặt hình thức, đây là một quá trình quyết định Markov (MDP).
>
> Giả sử ta biết hàm cost-to-go tối ưu $Q^*(s,a)$, tức tổng cải thiện lớn nhất đạt được nếu đi nước $a$ rồi tối ưu tiếp. Khi đó quy tắc chọn nước tốt nhất rất đơn giản: chọn $\arg\max_a Q^*(s,a)$.
>
> $Q^*$ thỏa đệ quy Bellman quen thuộc của quy hoạch động:"

$$
Q^*(s,a) \;=\; \mathbb{E}\Big[\, r + \gamma \max_{a'} Q^*(s',a') \,\Big]
$$

> "Khó khăn là không gian trạng thái có $2^m$ lời giải, nên không thể giải đệ quy này chính xác. Đây là lời nguyền số chiều quen thuộc. RL vượt qua bằng hai xấp xỉ:
>
> 1. **Lấy mẫu thay vì quét hết trạng thái:** cho agent chạy để thu các chuyển tiếp $(s,a,r,s')$, rồi cập nhật dần
> $$Q(s,a) \leftarrow Q(s,a) + \alpha\,\big[\, r + \gamma\max_{a'}Q(s',a') - Q(s,a) \,\big].$$
> Đây là Q-learning, có thể xem như value iteration ngẫu nhiên (stochastic approximation).
> 2. **Xấp xỉ hàm thay vì lưu bảng:** biểu diễn $Q$ bằng một mạng nơ-ron có tham số $\theta$. Đây là DQN; trong ngôn ngữ của Bertsekas và Tsitsiklis thì gọi là neuro-dynamic programming.
>
> **Thăm dò:** với xác suất $\varepsilon$, agent chọn một nước ngẫu nhiên. Việc này tương tự đa dạng hóa trong metaheuristic: để không chỉ học từ những nước mà ước lượng hiện tại cho là tốt."

**Ý cần chốt:** $Q(s,\text{add } i)$ chính là một **điểm số cho việc thêm item $i$** có tính đến các bước sau. Đó đúng là thứ ADD\* cần.

---

### Slide III.A: Training first, then search

> "Em dùng RL theo cách rất giới hạn: mạng nơ-ron **không** thay thế bộ giải. Khung E2LS giữ nguyên hoàn toàn: chọn item để bỏ, tabu, quy tắc chấp nhận, restart. Mạng chỉ làm **một việc**: chấm điểm các ứng viên tại mỗi nút ADD\*.
>
> - **Pha offline, cho từng instance:** sinh lời giải xuất phát bằng GA, rồi cho agent chạy 1000 lượt, mỗi lượt 100 nước, để học $Q_\theta$.
> - **Pha online:** ADD\* xếp hạng ứng viên theo $Q_\theta(s,\text{add } i)$ thay cho $p_i/AW(S,i)$.
>
> Về phương pháp luận: vì chỉ thay đúng một thành phần, mọi khác biệt về kết quả đều có thể quy cho hàm đánh giá. Em sẽ so sánh ở **cùng số bước** local search, vì gọi mạng tốn thời gian hơn nhiều so với tính tỉ lệ.
>
> Một chi tiết nhỏ: mạng có một đầu vào là khoảng cách tới incumbent (gap). Trong ADD\*, em đặt giá trị này bằng 0, vì ADD\* chỉ thêm item."

$$
\text{bài báo: } \operatorname{rank}(i) = \frac{p_i}{AW(S,i)}
\qquad\longrightarrow\qquad
\text{của em: } \operatorname{rank}(i) = Q_\theta\big((x_S, 0),\ \text{add } i\big)
$$

---

### Slide III.B: GA generation of starting solutions

> "Mỗi lượt học cần một lời giải xuất phát. Em dùng GA nhị phân: quần thể 50, 100 thế hệ, chọn lọc tournament, lai ghép uniform, đột biến lật bit. Giải mã bằng repair: thêm các item có bit 1 theo thứ tự, bỏ qua item không vừa, nên mọi cá thể đều khả thi.
>
> Lý do dùng GA thì thầy sẽ thấy quen: **phân phối huấn luyện phải khớp với phân phối sử dụng**. Local search thật luôn làm việc quanh các lời giải gần đầy và chất lượng cao. Nếu agent chỉ học từ lời giải ngẫu nhiên, hàm $Q$ học được sẽ không chính xác ở đúng vùng mà nó được dùng. 1000 lượt chạy GA độc lập cho các điểm xuất phát vừa tốt vừa đa dạng."

---

### Slide III.C: DQN: state, actions, reward, training

> "Đây là cách em mô hình hóa việc chọn nước đi thành một MDP."

**Trạng thái:** $s=(x_S,\ g)$, với
$$g = \frac{P_{best} - P(S)}{\bar p}.$$

> "Gap $g$ là cần thiết vì phần thưởng phụ thuộc vào incumbent. Thiếu nó, cùng một lời giải có thể có giá trị khác nhau, tức trạng thái không chứa đủ thông tin (không có tính Markov)."

**Nước đi:** thêm hoặc bỏ một item, tổng cộng $2m$ nước.

> "Nước không khả thi bị loại. Nước đảo ngược một nước vừa đi bị cấm 30 bước; đây chính là **tabu theo thuộc tính** mà thầy biết."

**Phần thưởng:** chỉ thưởng khi tạo được incumbent mới,
$$r_t = \frac{\max\big(0,\ P(S_{t+1}) - P_{best}\big)}{\bar p}
\quad\Longrightarrow\quad
\sum_t r_t = \frac{P_{best}^{\text{cuối}} - P_{best}^{\text{đầu}}}{\bar p}.$$

> "Tổng phần thưởng một lượt chạy bằng đúng tổng cải thiện incumbent. Nghĩa là **mục tiêu học trùng với mục tiêu của local search**. Bỏ item không bị phạt, nên agent được phép đi qua lời giải tệ hơn, giống tabu search. Agent cũng không thể kiếm thưởng bằng cách đi vòng: bỏ rồi thêm lại cùng một item thì không tạo incumbent mới."

**Học:** hồi quy mạng $Q_\theta$ về giá trị đích theo Bellman,
$$y = r + \gamma \max_{a'} Q_{\bar\theta}(s',a'),
\qquad
\mathcal{L}(\theta) = \big(Q_\theta(s,a) - y\big)^2.$$

> "Hai kỹ thuật ổn định quá trình học:
> - **Target network** $Q_{\bar\theta}$: một bản sao 'đóng băng' của mạng, cập nhật mỗi 5 lượt. Giống value iteration: dùng ước lượng của vòng trước để tính vòng sau, tránh đuổi theo một đích đang di chuyển.
> - **Replay buffer** chứa $10^4$ chuyển tiếp: học từ các mẫu cũ được trộn ngẫu nhiên, để các mẫu không tương quan với nhau theo cùng một lượt chạy.
>
> Các tham số còn lại:
> - $\gamma=0.99$ cho tầm nhìn khoảng $1/(1-\gamma)=100$ bước, bằng đúng độ dài một lượt.
> - Thăm dò: ban đầu hoàn toàn ngẫu nhiên, $\varepsilon$ giảm 1% mỗi lượt; khi không ngẫu nhiên thì chọn theo $\operatorname{softmax}(Q/\tau)$ với $\tau=5$."

---

### Slide III.D: Graph neural network, message passing

> "Hàm xấp xỉ $Q_\theta$ cần hai tính chất:
> 1. **Hiểu cấu trúc item–element.** Giá trị của một item phụ thuộc vào việc các element của nó đã được phủ chưa, và được dùng chung với những item nào. Đó chính là thứ $AW$ đo, nhưng $AW$ chỉ đo ở mức một bước.
> 2. **Không phụ thuộc kích thước.** Cùng một mô hình phải dùng được cho $m$ từ 85 đến 1000.
>
> Mạng nơ-ron đồ thị đáp ứng cả hai. Instance được biểu diễn thành đồ thị hai phía: ma trận quan hệ item–element chính là ma trận kề, cộng thêm một nút toàn cục."

- **Đặc trưng đầu vào** là các đại lượng quen thuộc:
  - item: đã chọn chưa, profit, $AW$ khi thêm và khi bỏ, có vừa không;
  - element: weight, đã được phủ chưa, bị phủ đúng một lần hay không, tỉ lệ item đang phủ;
  - nút toàn cục: mức dùng sức chứa, sức chứa còn lại, gap.
- **Mỗi vòng truyền thông điệp** (lặp 3 vòng):
  1. element tổng hợp thông tin từ các item phủ nó, có tách riêng các item đã chọn;
  2. item tổng hợp từ các element của nó và từ nút toàn cục;
  3. nút toàn cục tổng hợp tất cả.

> "Có thể hình dung đây là một phép **'tính $AW$ mở rộng' được học**. Thay vì chỉ hỏi 'element này đã được phủ chưa', mạng tổng hợp thông tin từ các item lân cận cùng chia sẻ element. Cuối cùng, cùng một head được dùng cho mọi item để ra $Q(s,\text{add } i)$ và $Q(s,\text{remove } i)$, nên số tham số không phụ thuộc vào $m$ hay $n$."

---

### Chuyển sang phần kết quả

> "Toàn bộ pipeline đã chạy thông. DQN chưa được huấn luyện. Kết quả ở phần sau là của bộ giải dùng luật tỉ lệ, chạy một lần 30 giây; đó là baseline mà DQN cần vượt qua."

---

## 3. Những câu thầy có thể hỏi

**"Cái này khác gì một hàm đánh giá học được, hay một hyper-heuristic?"**
Đúng là cùng họ: *learned heuristics inside metaheuristics*, hay "learning to search". Điểm riêng của em: (1) học **cost-to-go nhiều bước** chứ không phải điểm một bước; (2) đặt đúng vào bước xếp hạng của ADD\*, nơi bài báo vẫn dùng heuristic viết tay.

**"Vì sao dùng RL mà không học có giám sát?"**
Với cả quá trình tìm kiếm, không có sẵn nhãn "nước đi tốt nhất"; RL học từ tín hiệu cải thiện incumbent. Riêng quyết định trong ADD\* thì có thể tạo nhãn chính xác bằng vét cạn trên instance nhỏ. Em đã cài phương án này (`train_ranker.py`) để so sánh hai cách.

**"DQN học nước thêm/bỏ đơn lẻ, sao dùng cho ADD\* được?"**
$Q(s,\text{add } i)$ đo giá trị của việc thêm $i$ rồi tiếp tục tối ưu, nên khá gần câu hỏi ADD\* đặt ra: nên thêm item nào trước. Nhưng hai thứ không trùng khớp hoàn toàn, vì DQN học trên quỹ đạo của chính nó chứ không phải trên cây ADD\*. Đây là giả thuyết em cần kiểm chứng bằng thực nghiệm.

**"Có đảm bảo hội tụ không?"**
Q-learning dạng bảng hội tụ về $Q^*$ dưới các điều kiện chuẩn: mọi cặp trạng thái–nước đi được thăm vô hạn lần, bước học giảm dần (Watkins & Dayan, 1992). Khi xấp xỉ bằng mạng nơ-ron thì không còn đảm bảo này. Vì vậy phương pháp được đánh giá bằng thực nghiệm, giống như với metaheuristic.

**"Huấn luyện theo từng instance thì có tổng quát không?"**
Kiến trúc không phụ thuộc kích thước, nên về nguyên tắc có thể dùng chung. Hiện tại mỗi instance có một checkpoint riêng; huấn luyện chung trên nhiều instance là bước tiếp theo.

**"So sánh thế nào cho công bằng khi gọi mạng chậm hơn tính tỉ lệ?"**
So sánh ở cùng số bước local search và cùng số nút ADD\*; thời gian suy luận báo cáo riêng. Mạng chỉ được gọi khi có nhiều hơn $a_{num}$ ứng viên.

**"Vì sao đặt gap bằng 0 trong ADD\*?"**
ADD\* chỉ thêm item. Với gap bằng 0, mọi phần tăng profit đều được tính là phần thưởng ngay, nên $Q$ đo đúng lợi ích của việc thêm. Đây là một xấp xỉ.

**"Khi nào có kết quả?"**
Sau khi huấn luyện, em sẽ so sánh luật tỉ lệ với DQN ở cùng số bước, dùng 10 seed, trên Set I và Set II. Trọng tâm là Set II, nơi còn khoảng cách so với best-known.

---

## 4. Tài liệu nếu thầy muốn đọc thêm

- **R. S. Sutton, A. G. Barto**, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018. Sách nhập môn chuẩn.
- **D. P. Bertsekas, J. N. Tsitsiklis**, *Neuro-Dynamic Programming*, Athena Scientific, 1996. RL nhìn từ góc quy hoạch động; hợp với người làm tối ưu.
- **Y. Bengio, A. Lodi, A. Prouvost**, "Machine learning for combinatorial optimization: A methodological tour d'horizon", *European Journal of Operational Research* 290(2), 405–421, 2021. Tổng quan học máy cho tối ưu tổ hợp, viết cho cộng đồng OR.
- **V. Mnih et al.**, "Human-level control through deep reinforcement learning", *Nature* 518, 529–533, 2015. Bài gốc của DQN.
