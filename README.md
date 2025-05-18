# Tổng Hợp Các Thuật Toán Tìm Kiếm Trong Trí Tuệ Nhân Tạo

## 1. Mục tiêu

Repository này nhằm mục đích tổng hợp, tóm lược và minh họa các nhóm thuật toán tìm kiếm cơ bản và nâng cao được sử dụng trong lĩnh vực Trí tuệ Nhân tạo. Nội dung bao gồm định nghĩa, các thành phần cốt lõi, so sánh hiệu suất lý thuyết và thực tiễn (thông qua ví dụ trò chơi 8-puzzle), cũng như minh họa trực quan (nếu có) cho từng thuật toán.

## 2. Nội dung

### 2.1. Nhóm 1: Tìm Kiếm Không Có Thông Tin (Uninformed Search)

* **Tóm tắt nhóm:** Các thuật toán này duyệt không gian trạng thái mà không sử dụng bất kỳ thông tin bổ sung nào về bài toán ngoài định nghĩa của nó (trạng thái đầu, hàm chuyển, hàm kiểm tra đích). Chúng không "biết" trạng thái nào hứa hẹn hơn.
* **Các thuật toán chính:** BFS (Tìm kiếm theo chiều rộng), DFS (Tìm kiếm theo chiều sâu), UCS (Tìm kiếm chi phí thống nhất), IDDFS (Tìm kiếm sâu dần).

#### 2.1.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần:**
    1.  **Không gian trạng thái (State Space):** Tập hợp tất cả các trạng thái có thể của bài toán.
    2.  **Trạng thái ban đầu (Initial State):** Trạng thái bắt đầu của quá trình tìm kiếm.
    3.  **Hành động (Actions) & Hàm chuyển tiếp (Transition Model):** Các hành động có thể thực hiện từ một trạng thái và kết quả của chúng.
    4.  **Hàm mục tiêu (Goal Test):** Xác định xem một trạng thái có phải là đích hay không.
    5.  **Chi phí đường đi (Path Cost):** Giá trị số gán cho một đường đi.
* **Solution:** Một đường đi (chuỗi các hành động) từ trạng thái ban đầu đến trạng thái mục tiêu. Solution tối ưu là đường đi có chi phí thấp nhất.

#### 2.1.2. Hình ảnh GIF của từng thuật toán khi áp dụng lên trò chơi:
* *(Nhiều trình mô phỏng trực quan hóa quá trình các ô chữ di chuyển theo đường đi tìm được trên trò chơi 8-puzzle, giúp hình dung hoạt động của các thuật toán này.)*

![BFS](https://github.com/user-attachments/assets/674b51a8-5a7f-4acc-9192-680b14767b5c)
                                    **BFS (Breadth-First Search)**
                                    *(Duyệt từng lớp của cây tìm kiếm)*

![DFS](https://github.com/VoThanhNha/CS114.L21.KHCL/assets/115750000/680b6b35-e082-419d-94da-926f1351e7e7)
                                    **DFS (Depth-First Search)**
                                    *(Ưu tiên duyệt sâu theo một nhánh)*

![UCS](https://github.com/user-attachments/assets/46de1dd4-c1a5-4afd-bb39-2e1ddb1413c5)
                                    **UCS (Uniform Cost Search)**
                                    *(Mở rộng nút có tổng chi phí $g(n)$ thấp nhất)*

![IDDFS](https://github.com/user-attachments/assets/f7b534d7-9d72-46d4-ab32-aeaa6d71ffa0)
                                    **IDDFS (Iterative Deepening DFS)**
                                    *(Lặp lại DFS với giới hạn độ sâu tăng dần)*

#### 2.1.3. Hình ảnh so sánh hiệu suất của các thuật toán:
(Hiệu suất lý thuyết chung)
| Thuật Toán | Tính Đầy Đủ | Tính Tối Ưu           | Độ Phức Tạp Thời Gian | Độ Phức Tạp Không Gian |
| :--------- | :---------- | :-------------------- | :-------------------- | :--------------------- |
| BFS        | Có          | Có (nếu chi phí đều) | $O(b^d)$              | $O(b^d)$               |
| DFS        | Không       | Không                 | $O(b^m)$              | $O(bm)$                |
| UCS        | Có          | Có                    | $O(b^{1+C*/\epsilon})$ | $O(b^{1+C*/\epsilon})$  |
| IDDFS      | Có          | Có (nếu chi phí đều) | $O(b^d)$              | $O(bd)$                |
*(b: hệ số nhánh, d: độ sâu đích nông nhất, m: độ sâu max, C*: chi phí tối ưu, $\epsilon$: chi phí nhỏ nhất)*

![So sánh hiệu suất Nhóm 1](https://github.com/user-attachments/assets/0fde014b-1448-4cf4-adde-c1944496a842)
*Ghi chú về hiệu suất thực tế trên 8-puzzle:*
* BFS, UCS, IDDFS thường tìm ra lời giải tối ưu (ví dụ, 9 bước) một cách nhất quán cho các bài toán 8-puzzle điển hình.
* DFS thường gặp khó khăn, có thể timeout hoặc không tìm ra giải pháp trong thời gian hợp lý cho không gian tìm kiếm của 8-puzzle nếu không có các biện pháp cắt tỉa hoặc giới hạn độ sâu hiệu quả.

#### 2.1.4. Nhận xét về hiệu suất của các thuật toán trong nhóm này khi áp dụng lên trò chơi 8 ô chữ:
* **BFS và UCS:** Đảm bảo tìm ra lời giải ngắn nhất (tối ưu về số bước) do chi phí mỗi nước đi là 1. Tuy nhiên, chúng có thể tiêu tốn rất nhiều bộ nhớ ($O(b^d)$) khi độ sâu của lời giải tăng lên. Quan sát thực tế cho thấy chúng giải nhanh các bài toán 8-puzzle đơn giản.
* **DFS:** Yêu cầu bộ nhớ ít hơn nhiều ($O(bm)$), nhưng không đảm bảo tìm ra lời giải (có thể bị kẹt trong nhánh vô hạn hoặc rất dài) và nếu tìm ra cũng không chắc là tối ưu. Khi áp dụng cho 8-puzzle, DFS không giới hạn có thể dễ dàng bị lạc hoặc hết thời gian.
* **IDDFS:** Là một sự kết hợp tốt giữa BFS và DFS. Nó đầy đủ, tối ưu (với chi phí bước đều) và có yêu cầu bộ nhớ tốt hơn BFS ($O(bd)$). Trong thực tế, IDDFS giải được bài toán 8-puzzle một cách hiệu quả.

---

### 2.2. Nhóm 2: Tìm Kiếm Có Thông Tin (Informed Search / Heuristic Search)

* **Tóm tắt nhóm:** Sử dụng hàm "heuristic" (hàm ước lượng chi phí từ trạng thái hiện tại đến đích) để hướng dẫn quá trình tìm kiếm, giúp tìm kiếm hiệu quả hơn bằng cách ưu tiên các hướng đi "có vẻ" tốt hơn.
* **Các thuật toán chính:** Greedy Search (Tìm kiếm tham lam), A\* (A-star), IDA\* (Iterative Deepening A\*).

#### 2.2.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần:** Tương tự Nhóm 1, nhưng bổ sung thêm **Hàm Heuristic $h(n)$**. Ví dụ phổ biến cho 8-puzzle là `manhattan_distance` (khoảng cách Manhattan).
* **Solution:** Một đường đi từ trạng thái ban đầu đến đích. A\* và IDA\* đảm bảo giải pháp tối ưu nếu hàm heuristic là "chấp nhận được" (admissible - không đánh giá quá cao chi phí thực tế).

#### 2.2.2. Hình ảnh GIF của từng thuật toán khi áp dụng lên trò chơi:
* *(Các trình mô phỏng thường có chức năng hoạt ảnh để minh họa đường đi tìm được của các thuật toán này trên 8-puzzle.)*

![Greedy Search](https://github.com/user-attachments/assets/206ba3ce-3d9f-4453-8e73-5ff53d6b99f6)
                                    **Greedy Search (Tìm kiếm tham lam)**
                                    *(Luôn chọn nút có $h(n)$ nhỏ nhất)*

![A* Search](https://github.com/user-attachments/assets/6b497924-dd23-4c81-8fa4-6bbd3b9ce40e)
                                    **A\* Search (A-star)**
                                    *(Chọn nút có $f(n) = g(n) + h(n)$ nhỏ nhất)*

![IDA* Search](https://github.com/user-attachments/assets/4d135861-db2a-49fa-8e65-033a84c186e4)
                                    **IDA\* (Iterative Deepening A\*)**
                                    *(A\* với giới hạn độ sâu lặp dựa trên ngưỡng f_cost)*

#### 2.2.3. Hình ảnh so sánh hiệu suất của các thuật toán:
(Hiệu suất lý thuyết chung, phụ thuộc vào chất lượng heuristic)
| Thuật Toán    | Tính Đầy Đủ | Tính Tối Ưu                     | Độ Phức Tạp Thời Gian | Độ Phức Tạp Không Gian |
| :------------ | :---------- | :------------------------------- | :-------------------- | :--------------------- |
| Greedy        | Không       | Không                            | Phụ thuộc $h(n)$      | Phụ thuộc $h(n)$       |
| A\* | Có          | Có (nếu $h(n)$ chấp nhận được) | Phụ thuộc $h(n)$      | Thường là $O(b^d)$     |
| IDA\* | Có          | Có (nếu $h(n)$ chấp nhận được) | Phụ thuộc $h(n)$      | Thường là $O(bd)$      |

![So sánh hiệu suất Nhóm 2](https://github.com/user-attachments/assets/1d1a5a09-a816-4126-9f7b-bfa190633328)
*Ghi chú về hiệu suất thực tế trên 8-puzzle (sử dụng Manhattan distance):*
* A\* và IDA\* tỏ ra rất hiệu quả, luôn tìm ra đường đi tối ưu và nhanh chóng (ví dụ, nhiều trường hợp giải 9 bước chỉ trong khoảng thời gian rất ngắn, thường dưới 0.01 giây).
* Greedy cũng rất nhanh nhưng không phải lúc nào cũng tối ưu (ví dụ, có trường hợp Greedy tìm ra đường đi dài hơn đáng kể so với A\*/IDA\* cho cùng một bài toán).

#### 2.2.4. Nhận xét về hiệu suất trên trò chơi 8 ô chữ:
* **Greedy Search:** Rất nhanh do chỉ quan tâm đến heuristic $h(n)$. Tuy nhiên, điều này có thể dẫn đến việc chọn đường đi không tối ưu về tổng số bước.
* **A\* (A_Star):** Với một heuristic tốt như Manhattan distance, A\* rất hiệu quả cho 8-puzzle. Nó đảm bảo tìm ra đường đi ngắn nhất và thường nhanh hơn nhiều so với các thuật toán không có thông tin. Các quan sát thực tế cho thấy A\* liên tục tìm ra các giải pháp tối ưu với thời gian rất ngắn.
* **IDA\*:** Kết hợp tính tối ưu của A\* với ưu điểm về bộ nhớ của tìm kiếm sâu dần. Trên 8-puzzle, IDA\* cũng rất hiệu quả, tìm ra giải pháp tối ưu và thường có thời gian thực thi tương đương hoặc thậm chí nhanh hơn A\* một chút trong một số trường hợp, đồng thời về lý thuyết sử dụng ít bộ nhớ hơn.

---

### 2.3. Nhóm 3: Tìm Kiếm Cục Bộ (Local Search)

* **Tóm tắt nhóm:** Các thuật toán này bắt đầu từ một giải pháp tiềm năng và lặp đi lặp lại việc di chuyển đến các giải pháp "lân cận" để cố gắng cải thiện chất lượng giải pháp dựa trên một hàm mục tiêu. Thường không quan tâm đến đường đi từ trạng thái ban đầu.
* **Các thuật toán chính:** Hill Climbing (Leo đồi - gồm Simple, Steepest-Ascent, Stochastic), Simulated Annealing (Ủ mô phỏng), Genetic Algorithms (Thuật toán di truyền), Beam Search (Tìm kiếm chùm tia).

#### 2.3.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần:**
    1.  **Không gian trạng thái:** Tập hợp tất cả các cấu hình giải pháp có thể.
    2.  **Hàm mục tiêu (Objective Function) / Hàm đánh giá (Evaluation Function):** Đo lường "chất lượng" hoặc "chi phí" của một trạng thái/giải pháp. Đối với 8-puzzle, `manhattan_distance` thường được dùng làm hàm đánh giá (mục tiêu là đưa heuristic này về 0).
    3.  **Hàng xóm (Neighborhood):** Các trạng thái có thể đạt được từ trạng thái hiện tại thông qua một thay đổi nhỏ (ví dụ, một nước đi của ô trống trong 8-puzzle).
* **Solution:** Một trạng thái đạt được (thường là một cực trị cục bộ hoặc toàn cục của hàm mục tiêu). Đối với GA, solution có thể là một chuỗi hành động tốt nhất được tìm thấy.

#### 2.3.2. Hình ảnh GIF của từng thuật toán khi áp dụng lên trò chơi:

![Simple Hill Climbing](https://github.com/user-attachments/assets/54275283-6e4d-4a95-9855-3ab9328524cd)
![Steepest Ascent Hill Climbing](https://github.com/user-attachments/assets/fe72633e-71c8-40f7-9177-c91a9b42cb27)
![Stochastic Hill Climbing](https://github.com/user-attachments/assets/b1100bdf-4cb0-4a58-8161-153f1a0132d1)
**Hill Climbing (Simple, Steepest, Stochastic)**

![Simulated Annealing](https://github.com/user-attachments/assets/2e16612e-fad8-4977-9216-9a49bc8cda04)
**Simulated Annealing (SA)**

![Genetic Algorithms](https://github.com/user-attachments/assets/20d790d0-7812-462e-8019-448107f5900f)
**Genetic Algorithms (GA)**

![Beam Search](https://github.com/user-attachments/assets/c7a0af09-12e4-4de8-a0cf-c7cabe1a3644)
**Beam Search**
    *(Nhiều trình mô phỏng có chức năng hoạt ảnh để minh họa đường đi hoặc quá trình khám phá của các thuật toán này trên 8-puzzle.)*

#### 2.3.3. Hình ảnh so sánh hiệu suất của các thuật toán:
(Hiệu suất lý thuyết chung)
* **Hill Climbing:** Nhanh, đơn giản nhưng dễ bị kẹt ở cực trị địa phương (local optima) hoặc bình nguyên (plateau).
* **Simulated Annealing:** Có khả năng tìm ra cực trị toàn cục tốt hơn Hill Climbing, nhưng hiệu suất phụ thuộc vào "lịch trình làm nguội" (cooling schedule).
* **Genetic Algorithms:** Mạnh mẽ cho không gian tìm kiếm phức tạp, có thể tìm giải pháp gần tối ưu toàn cục. Yêu cầu điều chỉnh nhiều tham số.
* **Beam Search:** Là một sự cân bằng giữa tìm kiếm tham lam và BFS, hiệu suất phụ thuộc vào "độ rộng chùm tia" (beam width). Nếu beam width quá nhỏ, có thể giống Greedy; nếu quá lớn, có thể giống BFS.

![So sánh hiệu suất Nhóm 3](https://github.com/user-attachments/assets/aed3b324-8868-428f-8564-c0d474faf85d)
*Ghi chú về hiệu suất thực tế trên 8-puzzle:*
* Các biến thể Hill Climbing cho thấy hành vi này: Simple HC và Stochastic HC có lúc giải được, có lúc bị kẹt ở giá trị heuristic chưa tối ưu. Steepest HC có vẻ ổn định hơn và thường tìm được lời giải tốt hơn cho các bài toán 8-puzzle đơn giản.
* Simulated Annealing có thể tìm được đích nhưng quá trình "khám phá" (số trạng thái đã duyệt) thường rất dài.
* Beam Search (với beam width hợp lý) có thể tìm ra lời giải tốt cho 8-puzzle một cách hiệu quả.

#### 2.3.4. Nhận xét về hiệu suất trên trò chơi 8 ô chữ:
* **Hill Climbing (Simple, Steepest, Stochastic):** Các thuật toán này cố gắng giảm thiểu giá trị heuristic (ví dụ: Manhattan distance). Chúng rất nhanh nhưng dễ bị kẹt ở các trạng thái không phải là đích nhưng không có hàng xóm nào tốt hơn (cực trị địa phương).
* **Simulated Annealing (SA):** Có khả năng thoát khỏi các cực trị địa phương tốt hơn Hill Climbing. Khi áp dụng cho 8-puzzle, SA có thể tìm được trạng thái đích nhưng đường đi khám phá thường dài, phản ánh bản chất thăm dò của nó.
* **Genetic Algorithm (GA):** Có thể được dùng để tìm một chuỗi các nước đi giải 8-puzzle. Hiệu suất phụ thuộc vào các tham số như kích thước quần thể, số thế hệ, cách mã hóa giải pháp và các toán tử di truyền. Mục tiêu là tìm được chuỗi nước đi dẫn đến trạng thái đích hoặc trạng thái có heuristic tốt.
* **Beam Search:** Giữ một số lượng giới hạn trạng thái tốt nhất để mở rộng tiếp. Đây là một dạng giới hạn của BFS, giúp kiểm soát bộ nhớ nhưng có thể mất tính đầy đủ hoặc tối ưu nếu độ rộng chùm tia quá nhỏ. Cho 8-puzzle, với độ rộng chùm tia phù hợp, nó có thể hoạt động tốt.

---

### 2.4. Nhóm 4: Tìm Kiếm Trong Môi Trường Phức Tạp

* **Tóm tắt nhóm:** Các thuật toán và phương pháp này được thiết kế cho các bài toán mà môi trường có các đặc điểm như tính đối kháng (trong trò chơi), thông tin không đầy đủ (quan sát được một phần), hoặc môi trường động/không xác định.
* **Các thuật toán/khái niệm chính:** Tree Search AND-OR (Cây tìm kiếm AND-OR), Partially Observable environments (Môi trường quan sát được một phần), Unknown or Dynamic environments (Môi trường không xác định/động).

#### 2.4.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần:**
    * **AND-OR Tree Search:** Bài toán được phân rã thành các bài toán con (subproblems). Nút AND yêu cầu tất cả các bài toán con phải được giải quyết; nút OR chỉ cần một bài toán con được giải quyết.
    * **Partially Observable:** Agent không quan sát được đầy đủ trạng thái môi trường. Thay vào đó, agent duy trì một "trạng thái niềm tin" (belief state) - tập hợp các trạng thái có thể mà môi trường đang ở trong đó.
    * **Unknown/Dynamic Environment:** Agent có thể không biết trước đầy đủ về môi trường hoặc môi trường có thể thay đổi trong quá trình hoạt động. Agent cần học hoặc lập kế hoạch lại.
* **Solution:**
    * **AND-OR Tree Search:** Một "cây giải pháp" (solution graph/tree) thể hiện cách giải quyết bài toán gốc thông qua các bài toán con.
    * **Partially Observable:** Thường là một "chính sách" (policy) ánh xạ từ trạng thái niềm tin sang hành động.
    * **Unknown/Dynamic Environment:** Một kế hoạch có điều kiện (contingent plan) hoặc một chính sách thích ứng với sự thay đổi.

#### 2.4.2. Hình ảnh GIF của từng thuật toán/khái niệm khi áp dụng lên trò chơi:
![AND-OR Tree Search](https://github.com/user-attachments/assets/cf2fa2c9-db2b-4482-8b52-6d2387001bd4)
* **AND-OR Tree Search**

![Partially Observable Environment](https://github.com/user-attachments/assets/98117603-0325-4257-9003-f73205ab9a26)
* **Partially Observable**

![Unknown/Dynamic Environment](https://github.com/user-attachments/assets/250e3b65-9e2b-4d23-bbb9-591c8d20e806)
* **Unknown/Dynamic Environment**
    *(Các kịch bản trong những môi trường này thường được minh họa bằng các agent tự hành hoặc các hệ thống giải quyết vấn đề tương tác.)*

#### 2.4.3. Hình ảnh so sánh hiệu suất của các thuật toán/kịch bản:
* **AND-OR Trees:** Hiệu quả phụ thuộc vào cấu trúc phân rã của bài toán. Có thể giảm đáng kể không gian tìm kiếm nếu các bài toán con độc lập hoặc có thể giải quyết hiệu quả.
* **Partially Observable:** Thường phức tạp hơn đáng kể so với môi trường quan sát được hoàn toàn do phải làm việc với không gian của các trạng thái niềm tin, có thể rất lớn.
* **Unknown/Dynamic Environments:** Đòi hỏi thuật toán có khả năng học hỏi (ví dụ: Reinforcement Learning) hoặc lập kế hoạch lại (re-planning) khi có thông tin mới hoặc môi trường thay đổi.
    *(Hiệu suất trong các môi trường này rất đa dạng và phụ thuộc nhiều vào bản chất cụ thể của sự không chắc chắn hoặc tính động của môi trường.)*

#### 2.4.4. Nhận xét về hiệu suất trên trò chơi 8 ô chữ:
* **AND-OR Tree Search:** Có thể áp dụng cho 8-puzzle bằng cách xem việc đặt mỗi ô số vào đúng vị trí là một mục tiêu con (subgoal). Ví dụ, "đặt ô 1 vào vị trí (0,0) AND đặt ô 2 vào vị trí (0,1) AND...". Một cách tiếp cận là giải quyết tuần tự việc đưa từng ô về đúng vị trí, sử dụng một thuật toán tìm kiếm (như A\*) để giải quyết mỗi mục tiêu con đó trong khi giữ các ô đã đúng vị trí không bị xáo trộn.
* **Partially Observable:** Một biến thể của 8-puzzle có thể là agent chỉ nhìn thấy một vài ô, hoặc chỉ biết thông tin về một vùng nhất định (ví dụ, hàng đầu tiên luôn cố định và quan sát được, còn các ô khác thì agent duy trì một tập các trạng thái niềm tin). Việc tìm giải pháp trở nên khó khăn hơn nhiều, và các hành động có thể dựa trên việc cập nhật trạng thái niềm tin để giảm sự không chắc chắn.
* **Unknown/Dynamic (Non-Observable):** Nếu các quy tắc di chuyển của 8-puzzle ban đầu không rõ ràng, hoặc nếu có yếu tố bên ngoài làm thay đổi trạng thái (ví dụ, một ô ngẫu nhiên bị tráo đổi không theo hành động của agent). Trong trường hợp này, agent cần có khả năng khám phá, học luật của trò chơi, hoặc thích ứng với các thay đổi. Một kịch bản có thể là nhiều bản sao của 8-puzzle (có thể khác nhau) cùng phản ứng với một chuỗi hành động chung, mô phỏng agent hành động với kiến thức hạn chế hoặc khi cần một chiến lược chung cho nhiều tình huống tương tự.

---

### 2.5. Nhóm 5: Tìm Kiếm Trong Môi Trường Có Ràng Buộc (Constraint Satisfaction Problems - CSPs)

* **Tóm tắt nhóm:** Các thuật toán này được thiết kế để tìm các giải pháp thỏa mãn một tập hợp các ràng buộc hoặc điều kiện cho trước.
* **Các thuật toán chính:** Backtracking Search, Forward Checking, AC-3 (Arc Consistency Algorithm 3), Min-Conflicts.

#### 2.5.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần:**
    1.  **Tập biến (Variables):** $X = \{X_1, ..., X_n\}$.
    2.  **Tập miền giá trị (Domains):** $D = \{D_1, ..., D_n\}$, mỗi $D_i$ chứa các giá trị mà biến $X_i$ có thể nhận.
    3.  **Tập ràng buộc (Constraints):** Các quy tắc xác định những kết hợp giá trị nào của các biến là hợp lệ.
* **Solution:** Một phép gán giá trị cho tất cả các biến sao cho tất cả các ràng buộc đều được thỏa mãn. Nếu không có phép gán nào như vậy, bài toán không có lời giải.

#### 2.5.2. Hình ảnh GIF của từng thuật toán khi áp dụng lên bài toán:
![Backtracking Search](https://github.com/user-attachments/assets/6d5d64f6-ae37-429d-8d18-a4debdae6ab0)
* **Backtracking Search**

![Forward Checking](https://github.com/user-attachments/assets/e0c03cc8-db6f-44c9-90cc-af3e5d739151)
* **Forward Checking**

![AC-3](https://github.com/user-attachments/assets/55463b30-d3eb-4b02-bb01-2590030b1d15)
* **AC-3**

![Min-Conflicts](https://github.com/user-attachments/assets/20d3dc93-e43b-46af-bad0-91f713838055)
* **Min-Conflicts**

#### 2.5.3. Hình ảnh so sánh hiệu suất của các thuật toán:
* **Backtracking Search:** Là thuật toán cơ bản, có thể rất chậm nếu không có các cải tiến.
* **Forward Checking:** Cải thiện Backtracking bằng cách kiểm tra và loại bỏ các giá trị không tương thích ở các biến chưa được gán mỗi khi một biến được gán.
* **AC-3 (và các thuật toán nhất quán cung khác):** Thường được dùng để tiền xử lý hoặc xen kẽ với backtracking để giảm kích thước miền giá trị của các biến, từ đó giảm không gian tìm kiếm.
* **Min-Conflicts:** Một thuật toán tìm kiếm cục bộ cho CSPs. Nó bắt đầu với một phép gán đầy đủ nhưng có thể vi phạm ràng buộc, sau đó lặp đi lặp lại việc chọn một biến đang vi phạm ràng buộc và gán cho nó giá trị làm giảm thiểu số lượng ràng buộc bị vi phạm. Rất hiệu quả cho một số loại bài toán, đặc biệt là khi có thể tìm thấy giải pháp nhanh chóng, nhưng có thể bị kẹt ở cực trị địa phương.
* Hiệu suất của các thuật toán CSP phụ thuộc nhiều vào cấu trúc của bài toán, thứ tự gán biến, thứ tự thử giá trị và "độ chặt" của các ràng buộc.

#### 2.5.4. Nhận xét về hiệu suất trên trò chơi 8 ô chữ:
* 8-puzzle có thể được mô hình hóa như một CSP (ví dụ: mỗi ô trên bảng là một biến, miền giá trị là các số từ 0-8, ràng buộc là các số không được trùng nhau và phải tạo thành trạng thái đích).
* Tuy nhiên, việc giải 8-puzzle dưới dạng tìm đường đi (sequence of moves) thường hiệu quả hơn khi sử dụng các thuật toán tìm kiếm trạng thái (Nhóm 1, 2) thay vì các kỹ thuật CSP thuần túy.
* Các thuật toán CSP mạnh hơn cho các bài toán như tô màu đồ thị, Sudoku, lập lịch, nơi mục tiêu là tìm một phép gán giá trị thỏa mãn ràng buộc.
* **Min-Conflicts:** Tương tự như các thuật toán CSP khác, Min-Conflicts thường không phải là cách tiếp cận tự nhiên hoặc hiệu quả nhất để tìm *đường đi* giải quyết 8-puzzle. Nếu 8-puzzle được mô hình hóa như một CSP (tìm một cấu hình đích thỏa mãn ràng buộc), Min-Conflicts có thể được sử dụng để tìm một trạng thái đích. Tuy nhiên, nó không cung cấp đường đi từ trạng thái ban đầu và, giống như các thuật toán tìm kiếm cục bộ khác, có thể bị kẹt ở các cấu hình không phải là đích nếu hàm mục tiêu (số ràng buộc bị vi phạm) dẫn đến cực trị địa phương.

---

### 2.6. Nhóm 6: Học Tăng Cường (Reinforcement Learning - RL)

* **Tóm tắt nhóm:** Agent học cách hành động trong một môi trường để tối đa hóa một tín hiệu phần thưởng (reward) tích lũy nào đó, thường thông qua quá trình thử và sai.
* **Thuật toán ví dụ:** Q-Learning.

#### 2.6.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần:**
    1.  **Agent:** Thực thể học và ra quyết định.
    2.  **Environment (Môi trường):** Thế giới mà agent tương tác.
    3.  **States (S - Trạng thái):** Các tình huống mà agent có thể gặp phải.
    4.  **Actions (A - Hành động):** Các lựa chọn mà agent có thể thực hiện.
    5.  **Reward (R - Phần thưởng):** Tín hiệu phản hồi từ môi trường sau mỗi hành động, cho biết hành động đó tốt hay xấu.
    6.  **Policy ($\pi$ - Chính sách):** Chiến lược của agent, ánh xạ từ trạng thái sang hành động.
* **Solution:** Mục tiêu của RL là tìm ra một **chính sách tối ưu ($\pi\*$)** – một cách hành động giúp agent thu được tổng phần thưởng lớn nhất có thể trong dài hạn.

#### 2.6.2. Hình ảnh GIF của thuật toán (ví dụ Q-Learning):
* Tưởng tượng một robot học cách di chuyển trong một mê cung để đến được đích. Ban đầu nó di chuyển ngẫu nhiên, nhưng dần dần "học" được những đường đi tốt hơn dựa trên phần thưởng (ví dụ, đến gần đích hơn) hoặc phạt (ví dụ, đi vào ngõ cụt). Các "giá trị Q" (Q-values) của các cặp (trạng thái, hành động) được cập nhật liên tục. Tìm "Q-learning visualization gif" hoặc "Reinforcement learning grid world gif".
    *(Việc chạy các episodes huấn luyện Q-Learning cho 8-puzzle có thể được xem như quá trình agent học hỏi này.)*
    *(Hiện tại chưa có GIF cụ thể cho Q-Learning giải 8-puzzle trong repository này, nhưng bạn có thể tìm các minh họa tương tự trên mạng.)*

#### 2.6.3. Hình ảnh so sánh hiệu suất của các thuật toán (trong RL nói chung):
* Hiệu suất trong RL thường được đánh giá qua:
    * **Tốc độ hội tụ:** Mất bao lâu để học được chính sách tốt.
    * **Chất lượng chính sách cuối cùng:** Chính sách có gần tối ưu không.
    * **Sample Efficiency:** Cần bao nhiêu tương tác với môi trường để học.
    * **Khả năng xử lý không gian trạng thái/hành động lớn.**
* **Q-Learning:** Là một thuật toán model-free (không cần mô hình môi trường) và off-policy. Các tham số như tốc độ học ($\alpha$), hệ số chiết khấu ($\gamma$), tỷ lệ khám phá ($\epsilon$) và số lượt huấn luyện (episodes) ảnh hưởng lớn đến hiệu suất.
*Ghi chú về hiệu suất thực tế của Q-Learning trên 8-puzzle:*
* Hiệu suất thay đổi đáng kể. Có những lần Q-Learning tìm được giải pháp tối ưu rất nhanh (ví dụ, 2 bước chỉ trong khoảng 0.01 giây sau khi đã học, hoặc 9 bước trong khoảng 2 giây).
* Tuy nhiên, cũng có nhiều trường hợp chỉ tìm được "giải pháp một phần" (không đạt được trạng thái đích chính xác) hoặc đường đi dài hơn đáng kể, ngay cả sau nhiều episodes huấn luyện. Điều này cho thấy sự phụ thuộc lớn vào quá trình huấn luyện và các tham số.

#### 2.6.4. Nhận xét về hiệu suất trên trò chơi 8 ô chữ:
* 8-puzzle có thể được mô hình hóa thành bài toán RL: trạng thái là cấu hình bàn cờ, hành động là di chuyển ô trống. Phần thưởng có thể được thiết kế để khuyến khích việc đạt được trạng thái đích và giảm thiểu số bước đi.
* Q-Learning sẽ cố gắng học một hàm Q-value (giá trị của việc thực hiện hành động trong một trạng thái) để suy ra chính sách tối ưu. Việc lưu và tải Q-table (bảng Q) cho phép tái sử dụng kiến thức đã học.
* Quan sát thực tế cho thấy Q-Learning *có khả năng* giải được 8-puzzle. Tuy nhiên, hiệu suất (độ dài đường đi, thời gian tìm ra giải pháp tối ưu sau khi học) không ổn định và thường kém hơn so với các thuật toán tìm kiếm có thông tin như A\* cho một lượt giải cụ thể của một bài toán đã biết. Sự thành công và chất lượng giải pháp của Q-Learning phụ thuộc rất nhiều vào quá trình training (số episodes, các siêu tham số, hàm phần thưởng). RL nói chung và Q-Learning nói riêng thường mạnh hơn trong các môi trường động, không chắc chắn, hoặc khi cần học một chính sách tổng quát có thể áp dụng cho nhiều biến thể của bài toán, thay vì chỉ tìm một lời giải tối ưu cho một bài toán tĩnh cụ thể.
