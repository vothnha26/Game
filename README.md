# Tổng Hợp Các Thuật Toán Tìm Kiếm Trong Trí Tuệ Nhân Tạo

## 1. Mục tiêu

Dự án này nhằm mục đích tổng hợp, tóm lược và minh họa các nhóm thuật toán tìm kiếm cơ bản và nâng cao được sử dụng trong lĩnh vực Trí tuệ Nhân tạo. Nội dung bao gồm định nghĩa, các thành phần cốt lõi, so sánh hiệu suất lý thuyết và thực tiễn thông qua trò chơi 8-puzzle, cũng như minh họa trực quan (nếu có) cho từng thuật toán. Các triển khai cụ thể và kịch bản thử nghiệm được cung cấp, với kết quả được ghi lại một phần trong tệp log.

![UI_group1236](https://github.com/user-attachments/assets/496c135f-23ac-4f95-82c4-39b90e67be4e)

![UI_group4](https://github.com/user-attachments/assets/3cb14f09-4311-45ad-bb7f-3b08192cca9b)

![UI_group5](https://github.com/user-attachments/assets/1c1b9e49-fbc8-44d7-8e3b-f89a705a5601)

## 2. Nội dung

### 2.1. Nhóm 1: Tìm Kiếm Không Có Thông Tin (Uninformed Search)

* **Tóm tắt nhóm:** Các thuật toán này duyệt không gian trạng thái mà không sử dụng bất kỳ thông tin bổ sung nào về bài toán ngoài định nghĩa của nó (trạng thái đầu, hàm chuyển, hàm kiểm tra đích). Chúng không "biết" trạng thái nào hứa hẹn hơn.
* **Các thuật toán chính (được triển khai cho 8-puzzle):** BFS (Tìm kiếm theo chiều rộng), DFS (Tìm kiếm theo chiều sâu), UCS (Tìm kiếm chi phí thống nhất), IDDFS (Tìm kiếm sâu dần).

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

![DFS](https://github.com/user-attachments/assets/73525ad3-2633-4b72-a1e5-0d3793ca59ac)
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
| Thuật Toán | Tính Đầy Đủ | Tính Tối Ưu            | Độ Phức Tạp Thời Gian | Độ Phức Tạp Không Gian |
| :--------- | :---------- | :-------------------- | :-------------------- | :--------------------- |
| BFS        | Có          | Có (nếu chi phí đều) | $O(b^d)$              | $O(b^d)$               |
| DFS        | Không       | Không                 | $O(b^m)$              | $O(bm)$                |
| UCS        | Có          | Có                    | $O(b^{1+C*/\epsilon})$ | $O(b^{1+C*/\epsilon})$  |
| IDDFS      | Có          | Có (nếu chi phí đều) | $O(b^d)$              | $O(bd)$                |
*(b: hệ số nhánh, d: độ sâu đích nông nhất, m: độ sâu max, C*: chi phí tối ưu, $\epsilon$: chi phí nhỏ nhất)*

![So sánh hiệu suất Nhóm 1](https://github.com/user-attachments/assets/3d7fe063-9774-4600-b330-d4132b35641b)

*Ghi chú về hiệu suất thực tế trên 8-puzzle (tham khảo tệp log kết quả):*
* BFS, UCS, IDDFS thường tìm ra lời giải tối ưu một cách nhất quán. Thời gian chạy cho BFS và UCS thường nhanh, trong khi IDDFS có thể mất nhiều thời gian hơn do duyệt lại nhưng tiết kiệm bộ nhớ hơn BFS/UCS.
* DFS thường tìm ra lời giải không tối ưu và nhanh chóng hoặc có thể timeout/không tìm ra giải pháp trong thời gian hợp lý cho không gian tìm kiếm lớn nếu không có giới hạn độ sâu.

#### 2.1.4. Nhận xét về hiệu suất của các thuật toán trong nhóm này khi áp dụng lên trò chơi 8 ô chữ:
* **BFS và UCS:** Đảm bảo tìm ra lời giải ngắn nhất. Tuy nhiên, chúng có thể tiêu tốn nhiều bộ nhớ. Thực nghiệm cho thấy chúng giải quyết hiệu quả các bài toán 8-puzzle có độ sâu lời giải vừa phải.
* **DFS:** Yêu cầu bộ nhớ ít hơn nhiều, nhưng không đảm bảo tìm ra lời giải tối ưu. Khi áp dụng cho 8-puzzle, DFS không giới hạn thường không hiệu quả trong việc tìm lời giải tối ưu.
* **IDDFS:** Kết hợp ưu điểm của BFS và DFS. Trong thực tế, IDDFS giải được bài toán 8-puzzle một cách hiệu quả, tìm ra lời giải tối ưu mặc dù có thể tốn nhiều thời gian hơn BFS một chút.

---

### 2.2. Nhóm 2: Tìm Kiếm Có Thông Tin (Informed Search / Heuristic Search)

* **Tóm tắt nhóm:** Sử dụng hàm "heuristic" để hướng dẫn quá trình tìm kiếm, giúp tìm kiếm hiệu quả hơn.
* **Các thuật toán chính (được triển khai cho 8-puzzle):** Greedy Search, A\*, IDA\*.

#### 2.2.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần:** Tương tự Nhóm 1, nhưng bổ sung thêm **Hàm Heuristic $h(n)$**.
* **Solution:** Một đường đi từ trạng thái ban đầu đến đích. A\* và IDA\* đảm bảo giải pháp tối ưu nếu hàm heuristic là "chấp nhận được".

#### 2.2.2. Hình ảnh GIF của từng thuật toán khi áp dụng lên trò chơi:

![Greedy Search](https://github.com/user-attachments/assets/206ba3ce-3d9f-4453-8e73-5ff53d6b99f6)
                                        **Greedy Search (Tìm kiếm tham lam)**

![A* Search](https://github.com/user-attachments/assets/6b497924-dd23-4c81-8fa4-6bbd3b9ce40e)
                                        **A\* Search (A-star)**

![IDA* Search](https://github.com/user-attachments/assets/4d135861-db2a-49fa-8e65-033a84c186e4)
                                        **IDA\* (Iterative Deepening A\*)**

#### 2.2.3. Hình ảnh so sánh hiệu suất của các thuật toán:
(Hiệu suất lý thuyết chung, phụ thuộc vào heuristic)
| Thuật Toán    | Tính Đầy Đủ | Tính Tối Ưu                     | Độ Phức Tạp Thời Gian | Độ Phức Tạp Không Gian |
| :------------ | :---------- | :------------------------------- | :-------------------- | :--------------------- |
| Greedy        | Không       | Không                            | Phụ thuộc $h(n)$      | Phụ thuộc $h(n)$       |
| A\* | Có          | Có (nếu $h(n)$ chấp nhận được) | Phụ thuộc $h(n)$      | Thường là $O(b^d)$     |
| IDA\* | Có          | Có (nếu $h(n)$ chấp nhận được) | Phụ thuộc $h(n)$      | Thường là $O(bd)$      |

![So sánh hiệu suất Nhóm 2](https://github.com/user-attachments/assets/305dbce2-6945-4e74-86a5-e4dbe9c659af)

*Ghi chú về hiệu suất thực tế trên 8-puzzle (sử dụng Manhattan distance, tham khảo tệp log kết quả):*
* A\* và IDA\* tỏ ra rất hiệu quả, thường xuyên tìm ra đường đi tối ưu rất nhanh và duyệt ít trạng thái.
* Greedy cũng rất nhanh nhưng không phải lúc nào cũng tối ưu.

#### 2.2.4. Nhận xét về hiệu suất trên trò chơi 8 ô chữ:
* **Greedy Search:** Rất nhanh nhưng có thể không tối ưu.
* **A\* (A_Star):** Với heuristic tốt, A\* rất hiệu quả, đảm bảo tìm ra đường đi ngắn nhất và thường nhanh hơn nhiều so với các thuật toán không có thông tin.
* **IDA\*:** Kết hợp tính tối ưu của A\* với ưu điểm về bộ nhớ của tìm kiếm sâu dần, hoạt động hiệu quả trên 8-puzzle.

---

### 2.3. Nhóm 3: Tìm Kiếm Cục Bộ (Local Search)

* **Tóm tắt nhóm:** Bắt đầu từ một giải pháp tiềm năng và lặp đi lặp lại việc di chuyển đến các giải pháp "lân cận" để cải thiện .
* **Các thuật toán chính (được triển khai cho 8-puzzle):** Hill Climbing (Simple, Steepest-Ascent, Stochastic), Simulated Annealing, Genetic Algorithms, Beam Search.

#### 2.3.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần:** Không gian trạng thái, Hàm mục tiêu/đánh giá, Hàng xóm.
* **Solution:** Một trạng thái đạt được (thường là cực trị cục bộ/toàn cục). Đối với GA, có thể là một chuỗi hành động tốt nhất.

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

#### 2.3.3. Hình ảnh so sánh hiệu suất của các thuật toán:
(Hiệu suất lý thuyết chung)
* **Hill Climbing:** Nhanh, đơn giản nhưng dễ bị kẹt.
* **Simulated Annealing:** Có khả năng tìm cực trị toàn cục tốt hơn, phụ thuộc "lịch trình làm nguội".
* **Genetic Algorithms:** Mạnh mẽ cho không gian phức tạp, có thể tìm giải pháp gần tối ưu.
* **Beam Search:** Cân bằng giữa Greedy và BFS, phụ thuộc "độ rộng chùm tia".

![So sánh hiệu suất Nhóm 3](https://github.com/user-attachments/assets/ec7bc478-a931-4c55-b00e-d92f722dc74d)

Test case: [[1, 2, 3], [4, 0, 5], [6, 7, 8]]
*Ghi chú về hiệu suất thực tế trên 8-puzzle (tham khảo tệp log kết quả):*
* Các biến thể Hill Climbing: Simple/Stochastic HC có thể bị kẹt; Steepest HC ổn định hơn.
* Simulated Annealing: Có thể tìm được đích nhưng quá trình khám phá thường dài.
* Genetic Algorithm: Có thể tìm được lời giải tối ưu sau một số thế hệ, thời gian chạy có thể đáng kể.
* Beam Search: Với beam width hợp lý, có thể tìm ra lời giải tối ưu hiệu quả và nhanh chóng.

#### 2.3.4. Nhận xét về hiệu suất trên trò chơi 8 ô chữ:
* **Hill Climbing:** Rất nhanh nhưng dễ bị kẹt ở cực trị địa phương. Steepest Ascent HC có thể hiệu quả hơn.
* **Simulated Annealing (SA):** Có khả năng thoát khỏi cực trị địa phương, nhưng đường đi khám phá thường dài.
* **Genetic Algorithm (GA):** Có thể tìm ra lời giải tối ưu nhưng cần thời gian và số thế hệ nhất định.
* **Beam Search:** Với độ rộng chùm tia phù hợp, hoạt động tốt và nhanh chóng.

---

### 2.4. Nhóm 4: Tìm Kiếm Trong Môi Trường Phức Tạp

* **Tóm tắt nhóm:** Các thuật toán cho môi trường có tính đối kháng, thông tin không đầy đủ, hoặc động/không xác định.
* **Các thuật toán/khái niệm chính (minh họa bằng các ví dụ riêng):** Tree Search AND-OR, Partially Observable environments, Unknown or Dynamic environments.

#### 2.4.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần:**
    * **AND-OR Tree Search:** Phân rã bài toán thành các bài toán con.
    * **Partially Observable:** Agent duy trì "trạng thái niềm tin".
    * **Unknown/Dynamic Environment:** Agent cần học hoặc lập kế hoạch lại.
* **Solution:** Cây giải pháp (AND-OR), chính sách (Partially Observable), kế hoạch có điều kiện/chính sách thích ứng (Unknown/Dynamic).

#### 2.4.2. Hình ảnh GIF của từng thuật toán/khái niệm khi áp dụng lên trò chơi:
![AND-OR Tree Search](https://github.com/user-attachments/assets/cf2fa2c9-db2b-4482-8b52-6d2387001bd4)
* **AND-OR Tree Search**

![Partially Observable Environment](https://github.com/user-attachments/assets/98117603-0325-4257-9003-f73205ab9a26)
* **Partially Observable**

![Unknown/Dynamic Environment](https://github.com/user-attachments/assets/250e3b65-9e2b-4d23-bbb9-591c8d20e806)
* **Unknown/Dynamic Environment**
    *(Các kịch bản này được minh họa bằng các ví dụ như robot giao hàng, máy hút bụi, agent khám phá mê cung.)*

#### 2.4.3. Hình ảnh so sánh hiệu suất của các thuật toán/kịch bản:
* **AND-OR Trees:** Hiệu quả phụ thuộc cấu trúc phân rã.
* **Partially Observable:** Phức tạp hơn do không gian trạng thái niềm tin.
* **Unknown/Dynamic Environments:** Đòi hỏi khả năng học hỏi/lập kế hoạch lại.

#### 2.4.4. Nhận xét về hiệu suất và ứng dụng (bao gồm trò chơi 8 ô chữ):
* Các thuật toán này được minh họa trên các bài toán cụ thể (robot giao hàng, máy hút bụi, khám phá mê cung).
* **Đối với trò chơi 8 ô chữ (ứng dụng khái niệm):**
    * **AND-OR Tree Search:** Xem việc đặt mỗi ô là mục tiêu con.
    * **Partially Observable:** Agent chỉ thấy một phần bàn cờ.
    * **Unknown/Dynamic:** Agent cần học luật chơi hoặc thích ứng với thay đổi.

---

### 2.5. Nhóm 5: Tìm Kiếm Trong Môi Trường Có Ràng Buộc (Constraint Satisfaction Problems - CSPs)

* **Tóm tắt nhóm:** Tìm các giải pháp thỏa mãn một tập hợp các ràng buộc.
* **Các thuật toán chính (minh họa bằng các ví dụ riêng):** Backtracking Search, Min-Conflicts.

#### 2.5.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần:** Tập biến, Tập miền giá trị, Tập ràng buộc.
* **Solution:** Một phép gán giá trị cho tất cả các biến sao cho tất cả các ràng buộc đều được thỏa mãn.

#### 2.5.2. Hình ảnh GIF của từng thuật toán khi áp dụng lên bài toán:
![Backtracking Search](https://github.com/user-attachments/assets/6d5d64f6-ae37-429d-8d18-a4debdae6ab0)
* **Backtracking Search**

![Forward Checking](https://github.com/user-attachments/assets/e0c03cc8-db6f-44c9-90cc-af3e5d739151)
* **Forward Checking (Kỹ thuật liên quan)**

![AC-3](https://github.com/user-attachments/assets/55463b30-d3eb-4b02-bb01-2590030b1d15)
* **AC-3 (Kỹ thuật liên quan)**

![Min-Conflicts](https://github.com/user-attachments/assets/20d3dc93-e43b-46af-bad0-91f713838055)
* **Min-Conflicts**

#### 2.5.3. Hình ảnh so sánh hiệu suất của các thuật toán:
* **Backtracking Search:** Cơ bản, có thể chậm.
* **Forward Checking & AC-3:** Cải thiện Backtracking. Cụ thể AC-3 ở project dùng để khởi tạo ô 8 chữ
* **Min-Conflicts:** Thuật toán tìm kiếm cục bộ hiệu quả cho một số CSPs, có thể kẹt.
* Hiệu suất phụ thuộc cấu trúc bài toán và ràng buộc.

#### 2.5.4. Nhận xét về hiệu suất và ứng dụng (bao gồm trò chơi 8 ô chữ):
* Các thuật toán này được minh họa trên các bài toán CSP kinh điển (tô màu bản đồ, N-Queens).
* **Đối với trò chơi 8 ô chữ (ứng dụng khái niệm):**
    * Có thể mô hình hóa như CSP, nhưng giải dưới dạng tìm *đường đi* thường hiệu quả hơn.
    * **Min-Conflicts:** Có thể dùng để tìm cấu hình đích, nhưng không cung cấp đường đi.

---

### 2.6. Nhóm 6: Học Tăng Cường (Reinforcement Learning - RL)

* **Tóm tắt nhóm:** Agent học cách hành động để tối đa hóa phần thưởng tích lũy.
* **Thuật toán ví dụ:** Q-Learning.
* **Lưu ý:** Nhóm này hiện *chưa được triển khai*. Các nhận xét dưới đây mang tính lý thuyết.

#### 2.6.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần:** Agent, Môi trường, Trạng thái, Hành động, Phần thưởng, Chính sách.
* **Solution:** Tìm ra một **chính sách tối ưu ($\pi\*$)**.

#### 2.6.2. Hình ảnh GIF của thuật toán (ví dụ Q-Learning):
![qlearning](https://github.com/user-attachments/assets/284098bd-a219-45c5-a7bc-f6b5a93f18a7)

#### 2.6.3. Hình ảnh so sánh hiệu suất của các thuật toán (trong RL nói chung):
* Đánh giá qua: Tốc độ hội tụ,  chính sách, Sample Efficiency.
* **Q-Learning:** Model-free, off-policy. Tham số ảnh hưởng lớn đến hiệu suất.
*Ghi chú về hiệu suất thực tế của Q-Learning trên 8-puzzle (lý thuyết):*
* Hiệu suất có thể thay đổi, phụ thuộc vào huấn luyện và tham số.

#### 2.6.4. Nhận xét về hiệu suất trên trò chơi 8 ô chữ:
* 8-puzzle có thể mô hình hóa thành bài toán RL.
* Q-Learning cố gắng học hàm Q-value để suy ra chính sách tối ưu.
* Q-Learning *có khả năng* giải 8-puzzle, nhưng hiệu suất có thể không ổn định và thường kém hơn A\* cho một lượt giải cụ thể. RL mạnh hơn trong môi trường động/không chắc chắn.

---

## 3. Kết luận

* **Hệ thống hóa kiến thức:** Tổng hợp các nhóm thuật toán tìm kiếm AI chính.
* **Minh họa thực tiễn:** Triển khai và so sánh hiệu suất các thuật toán tìm kiếm trạng thái (Nhóm 1, 2, 3) trên bài toán 8-puzzle, đồng thời minh họa các khái niệm tìm kiếm nâng cao (Nhóm 4, 5) qua các ví dụ phù hợp.
* **Cung cấp tài liệu tham khảo:** Tạo ra một nguồn tài liệu học tập về các thuật toán tìm kiếm, giúp làm rõ ưu nhược điểm và tính phù hợp của từng nhóm cho các loại bài toán khác nhau.
* **Định hướng phát triển:** Đặt nền tảng cho việc khám phá các lĩnh vực liên quan như Học tăng cường (Nhóm 6).
