# Tổng Hợp Các Thuật Toán Tìm Kiếm Trong Trí Tuệ Nhân Tạo

## 1. Mục tiêu

Dự án này nhằm mục đích tổng hợp, tóm lược và minh họa các nhóm thuật toán tìm kiếm cơ bản và nâng cao được sử dụng trong lĩnh vực Trí tuệ Nhân Tạo. Nội dung bao gồm định nghĩa, các thành phần cốt lõi, so sánh hiệu suất lý thuyết và thực tiễn thông qua trò chơi 8-puzzle, cũng như minh họa trực quan (nếu có) cho từng thuật toán. Các triển khai cụ thể và kịch bản thử nghiệm được cung cấp, với kết quả được ghi lại một phần trong tệp log.

![UI_group1236](https://github.com/user-attachments/assets/496c135f-23ac-4f95-82c4-39b90e67be4e)
*Giao diện chính cho Nhóm 1, 2, 3 và 6*

![UI_group4](https://github.com/user-attachments/assets/3cb14f09-4311-45ad-bb7f-3b08192cca9b)
*Giao diện cho Nhóm 4: Tìm kiếm trong môi trường phức tạp*

![UI_group5](https://github.com/user-attachments/assets/1c1b9e49-fbc8-44d7-8e3b-f89a705a5601)
*Giao diện cho Nhóm 5: Tìm kiếm trong môi trường có ràng buộc*

## 2. Nội dung

### 2.1. Nhóm 1: Tìm Kiếm Không Có Thông Tin (Uninformed Search)

* **Tóm tắt nhóm:** Các thuật toán này duyệt không gian trạng thái mà không sử dụng bất kỳ thông tin bổ sung nào về bài toán ngoài định nghĩa của nó (trạng thái đầu, hàm chuyển, hàm kiểm tra đích). Chúng không "biết" trạng thái nào hứa hẹn hơn.
* **Các thuật toán chính (được triển khai cho 8-puzzle trong `main.py`):** BFS (Tìm kiếm theo chiều rộng), DFS (Tìm kiếm theo chiều sâu), UCS (Tìm kiếm chi phí thống nhất), IDDFS (Tìm kiếm sâu dần).

#### 2.1.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần (chung):**
    1.  **Không gian trạng thái (State Space):** Tập hợp tất cả các trạng thái có thể của bài toán.
    2.  **Trạng thái ban đầu (Initial State):** Trạng thái bắt đầu của quá trình tìm kiếm.
    3.  **Hành động (Actions) & Hàm chuyển tiếp (Transition Model):** Các hành động có thể thực hiện từ một trạng thái và kết quả của chúng.
    4.  **Hàm mục tiêu (Goal Test):** Xác định xem một trạng thái có phải là đích hay không.
    5.  **Chi phí đường đi (Path Cost):** Giá trị số gán cho một đường đi.
* **Solution (chung):** Một đường đi (chuỗi các hành động) từ trạng thái ban đầu đến trạng thái mục tiêu. Solution tối ưu là đường đi có chi phí thấp nhất.

* **Áp dụng cho 8-puzzle (trong `main.py`):**
    * **Trạng thái (State):** Được biểu diễn bằng một danh sách các danh sách (list of lists) 3x3, hoặc một tuple các tuple 3x3, thể hiện vị trí các số từ 0 đến 8 (0 là ô trống).
    * **Hành động & Hàm chuyển tiếp:** Di chuyển ô trống (số 0) lên, xuống, trái, hoặc phải đến ô kề cạnh. Hàm `get_neighbors(state)` tạo ra các trạng thái kế tiếp.
    * **Solution:** Một danh sách các trạng thái (list of states) từ trạng thái ban đầu đến trạng thái mục tiêu. Solution tối ưu là danh sách có ít trạng thái nhất (tương đương ít bước di chuyển nhất).

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

#### 2.1.3. So sánh hiệu suất lý thuyết của các thuật toán trong Nhóm 1:
*(b: hệ số nhánh (branching factor), d: độ sâu của lời giải nông nhất, m: độ sâu tối đa của không gian trạng thái, C*: chi phí của lời giải tối ưu, $\epsilon$: chi phí hành động nhỏ nhất, lớn hơn 0)*

* **BFS (Breadth-First Search):**
    * **Tính đầy đủ (Completeness):** Có (nếu b hữu hạn).
    * **Tính tối ưu (Optimality):** Có (nếu chi phí mỗi hành động là như nhau, ví dụ bằng 1).
    * **Độ phức tạp thời gian (Time Complexity):** $O(b^d)$.
    * **Độ phức tạp không gian (Space Complexity):** $O(b^d)$.

* **DFS (Depth-First Search):**
    * **Tính đầy đủ:** Không (có thể bị kẹt trong nhánh vô hạn nếu không gian trạng thái là vô hạn và không có kiểm tra lặp). Có nếu không gian trạng thái hữu hạn và không có chu trình.
    * **Tính tối ưu:** Không.
    * **Độ phức tạp thời gian:** $O(b^m)$ (có thể rất lớn).
    * **Độ phức tạp không gian:** $O(bm)$ (khá tốt).

* **UCS (Uniform Cost Search):**
    * **Tính đầy đủ:** Có (nếu chi phí mỗi hành động $\ge \epsilon > 0$).
    * **Tính tối ưu:** Có.
    * **Độ phức tạp thời gian:** $O(b^{1+\lfloor C*/\epsilon \rfloor})$.
    * **Độ phức tạp không gian:** $O(b^{1+\lfloor C*/\epsilon \rfloor})$.

* **IDDFS (Iterative Deepening DFS):**
    * **Tính đầy đủ:** Có (nếu b hữu hạn).
    * **Tính tối ưu:** Có (nếu chi phí mỗi hành động là như nhau).
    * **Độ phức tạp thời gian:** $O(b^d)$.
    * **Độ phức tạp không gian:** $O(bd)$.

![So sánh hiệu suất Nhóm 1](https://github.com/user-attachments/assets/3d7fe063-9774-4600-b330-d4132b35641b)
*Biểu đồ so sánh hiệu suất thực tế của Nhóm 1 trên 8-puzzle*

*Ghi chú về hiệu suất thực tế trên 8-puzzle (tham khảo tệp log kết quả `8_puzzle_test_log.txt`):*
* BFS, UCS, IDDFS thường tìm ra lời giải tối ưu một cách nhất quán. Thời gian chạy cho BFS và UCS thường nhanh, trong khi IDDFS có thể mất nhiều thời gian hơn do duyệt lại nhưng tiết kiệm bộ nhớ hơn BFS/UCS.
* DFS thường tìm ra lời giải không tối ưu và nhanh chóng hoặc có thể timeout/không tìm ra giải pháp trong thời gian hợp lý cho không gian tìm kiếm lớn nếu không có giới hạn độ sâu (trong code đã giới hạn `max_practical_dfs_depth`).

#### 2.1.4. Nhận xét về hiệu suất của các thuật toán trong nhóm này khi áp dụng lên trò chơi 8 ô chữ:
* **BFS và UCS:** Đảm bảo tìm ra lời giải ngắn nhất. Tuy nhiên, chúng có thể tiêu tốn nhiều bộ nhớ. Thực nghiệm cho thấy chúng giải quyết hiệu quả các bài toán 8-puzzle có độ sâu lời giải vừa phải.
* **DFS:** Yêu cầu bộ nhớ ít hơn nhiều, nhưng không đảm bảo tìm ra lời giải tối ưu. Khi áp dụng cho 8-puzzle, DFS không giới hạn thường không hiệu quả trong việc tìm lời giải tối ưu. Phiên bản giới hạn độ sâu thực tế trong code giúp tránh tình trạng duyệt vô hạn.
* **IDDFS:** Kết hợp ưu điểm của BFS và DFS. Trong thực tế, IDDFS giải được bài toán 8-puzzle một cách hiệu quả, tìm ra lời giải tối ưu mặc dù có thể tốn nhiều thời gian hơn BFS một chút.

---

### 2.2. Nhóm 2: Tìm Kiếm Có Thông Tin (Informed Search / Heuristic Search)

* **Tóm tắt nhóm:** Sử dụng hàm "heuristic" để hướng dẫn quá trình tìm kiếm, giúp tìm kiếm hiệu quả hơn.
* **Các thuật toán chính (được triển khai cho 8-puzzle trong `main.py`):** Greedy Search, A\*, IDA\*.

#### 2.2.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần (chung):** Tương tự Nhóm 1, nhưng bổ sung thêm **Hàm Heuristic $h(n)$**.
* **Solution (chung):** Một đường đi từ trạng thái ban đầu đến đích. A\* và IDA\* đảm bảo giải pháp tối ưu nếu hàm heuristic là "chấp nhận được" (admissible).

* **Áp dụng cho 8-puzzle (trong `main.py`):**
    * **Trạng thái (State):** Tương tự Nhóm 1 (ma trận 3x3 hoặc tuple các tuple).
    * **Hàm Heuristic $h(n)$:** Trong code, hàm `manhattan_distance(state, target_state)` được sử dụng, tính tổng khoảng cách Manhattan của các ô (trừ ô trống) đến vị trí đúng của chúng trong trạng thái đích.
    * **Solution:** Một danh sách các trạng thái từ trạng thái ban đầu đến trạng thái mục tiêu. Với A\* và IDA\* sử dụng heuristic Manhattan (là heuristic chấp nhận được cho 8-puzzle), solution tìm được đảm bảo là tối ưu (ít bước di chuyển nhất).

#### 2.2.2. Hình ảnh GIF của từng thuật toán khi áp dụng lên trò chơi:

![Greedy Search](https://github.com/user-attachments/assets/206ba3ce-3d9f-4453-8e73-5ff53d6b99f6)
                                        **Greedy Search (Tìm kiếm tham lam)**
                                        *(Luôn chọn trạng thái có heuristic $h(n)$ nhỏ nhất)*

![A* Search](https://github.com/user-attachments/assets/6b497924-dd23-4c81-8fa4-6bbd3b9ce40e)
                                        **A\* Search (A-star)**
                                        *(Chọn trạng thái có $f(n) = g(n) + h(n)$ nhỏ nhất)*

![IDA* Search](https://github.com/user-attachments/assets/4d135861-db2a-49fa-8e65-033a84c186e4)
                                        **IDA\* (Iterative Deepening A\*)**
                                        *(Lặp lại tìm kiếm giới hạn theo $f(n)$ tăng dần)*

#### 2.2.3. So sánh hiệu suất lý thuyết của các thuật toán trong Nhóm 2:
*(Hiệu suất phụ thuộc mạnh vào chất lượng của hàm heuristic $h(n)$)*

* **Greedy Best-First Search:**
    * **Tính đầy đủ:** Không (có thể bị kẹt trong vòng lặp nếu không kiểm tra trạng thái đã thăm). Có nếu không gian trạng thái hữu hạn và có kiểm tra lặp.
    * **Tính tối ưu:** Không.
    * **Độ phức tạp thời gian:** $O(b^m)$ trong trường hợp xấu nhất, nhưng có thể cải thiện đáng kể với heuristic tốt.
    * **Độ phức tạp không gian:** $O(b^m)$ trong trường hợp xấu nhất (lưu tất cả các nút trong danh sách mở).

* **A\* Search:**
    * **Tính đầy đủ:** Có (nếu chi phí mỗi hành động $\ge \epsilon > 0$ và số nút có $f \le C^*$ là hữu hạn).
    * **Tính tối ưu:** Có (nếu heuristic $h(n)$ là chấp nhận được (admissible - không đánh giá quá cao chi phí thực tế) và nhất quán (consistent/monotonic)).
    * **Độ phức tạp thời gian:** Số nút được mở rộng phụ thuộc vào chất lượng của heuristic. Có thể là $O(b^d)$ trong trường hợp tốt nhất (heuristic hoàn hảo) hoặc $O(b^{C^*/\epsilon})$ trong trường hợp xấu hơn.
    * **Độ phức tạp không gian:** Giữ tất cả các nút đã tạo trong bộ nhớ, thường là $O(b^d)$ hoặc tương đương với độ phức tạp thời gian.

* **IDA\* (Iterative Deepening A\*):**
    * **Tính đầy đủ:** Có (tương tự A\*).
    * **Tính tối ưu:** Có (tương tự A\*).
    * **Độ phức tạp thời gian:** Phụ thuộc vào số lượng giá trị f-cost riêng biệt và chất lượng heuristic. Có thể tương đương A\* nếu số lượng giá trị f-cost không quá nhiều.
    * **Độ phức tạp không gian:** $O(bd)$ (tương tự IDDFS).

![So sánh hiệu suất Nhóm 2](https://github.com/user-attachments/assets/305dbce2-6945-4e74-86a5-e4dbe9c659af)
*Biểu đồ so sánh hiệu suất thực tế của Nhóm 2 trên 8-puzzle*

*Ghi chú về hiệu suất thực tế trên 8-puzzle (sử dụng Manhattan distance, tham khảo tệp log kết quả `8_puzzle_test_log.txt`):*
* A\* và IDA\* tỏ ra rất hiệu quả, thường xuyên tìm ra đường đi tối ưu rất nhanh và duyệt ít trạng thái hơn nhiều so với Nhóm 1.
* Greedy cũng rất nhanh nhưng không phải lúc nào cũng tối ưu; đường đi tìm được có thể dài hơn.

#### 2.2.4. Nhận xét về hiệu suất trên trò chơi 8 ô chữ:
* **Greedy Search:** Rất nhanh nhưng có thể không tối ưu.
* **A\* (A_Star):** Với heuristic Manhattan, A\* rất hiệu quả, đảm bảo tìm ra đường đi ngắn nhất và thường nhanh hơn nhiều so với các thuật toán không có thông tin.
* **IDA\*:** Kết hợp tính tối ưu của A\* với ưu điểm về bộ nhớ của tìm kiếm sâu dần, hoạt động hiệu quả trên 8-puzzle, đặc biệt hữu ích cho các bài toán lớn hơn nơi A\* có thể gặp vấn đề bộ nhớ.

---

### 2.3. Nhóm 3: Tìm Kiếm Cục Bộ (Local Search)

* **Tóm tắt nhóm:** Bắt đầu từ một giải pháp tiềm năng (trạng thái hiện tại) và lặp đi lặp lại việc di chuyển đến các giải pháp "lân cận" để cải thiện dựa trên một hàm mục tiêu hoặc đánh giá.
* **Các thuật toán chính (được triển khai cho 8-puzzle trong `main.py`):** Hill Climbing (Simple, Steepest-Ascent, Stochastic), Simulated Annealing, Genetic Algorithms, Beam Search.

#### 2.3.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần (chung):** Không gian trạng thái, Hàm mục tiêu/đánh giá (thường là heuristic $h(n)$ để cực tiểu hóa), Hàng xóm (các trạng thái lân cận).
* **Solution (chung):** Một trạng thái đạt được (thường là cực trị cục bộ/toàn cục của hàm đánh giá). Không nhất thiết là một đường đi từ trạng thái ban đầu.

* **Áp dụng cho 8-puzzle (trong `main.py`):**
    * **Trạng thái (State):** Một cấu hình bàn cờ 3x3. Các thuật toán này thường chỉ làm việc với một hoặc một vài trạng thái hiện tại (ví dụ: quần thể trong GA).
    * **Hàm đánh giá:** Thường là hàm heuristic `manhattan_distance(state, target_state)` để đánh giá "chất lượng" của một trạng thái (càng gần đích, giá trị càng nhỏ).
    * **Hàng xóm (Neighbors):** Các trạng thái có thể đạt được từ trạng thái hiện tại bằng một hành động (di chuyển ô trống).
    * **Solution:**
        * Với Hill Climbing, SA, Beam Search: "Solution" là danh sách các trạng thái (đường đi) được duyệt qua để đến trạng thái cuối cùng (tốt nhất) mà thuật toán tìm được. Trạng thái cuối cùng này có thể là đích, cực tiểu cục bộ, hoặc trạng thái tốt nhất sau một số bước.
        * Với Genetic Algorithm (GA): "Cá thể" trong quần thể là một chuỗi các nước đi (`move_sequence`). "Solution" là đường đi (danh sách các trạng thái) được tạo ra bằng cách áp dụng chuỗi nước đi của cá thể tốt nhất lên trạng thái ban đầu.

#### 2.3.2. Hình ảnh GIF của từng thuật toán khi áp dụng lên trò chơi:

![Simple Hill Climbing](https://github.com/user-attachments/assets/54275283-6e4d-4a95-9855-3ab9328524cd)
![Steepest Ascent Hill Climbing](https://github.com/user-attachments/assets/fe72633e-71c8-40f7-9177-c91a9b42cb27)
![Stochastic Hill Climbing](https://github.com/user-attachments/assets/b1100bdf-4cb0-4a58-8161-153f1a0132d1)
**Hill Climbing (Simple, Steepest, Stochastic)**
*(Cố gắng di chuyển đến trạng thái láng giềng tốt hơn)*

![Simulated Annealing](https://github.com/user-attachments/assets/2e16612e-fad8-4977-9216-9a49bc8cda04)
**Simulated Annealing (SA)**
*(Cho phép bước đi "xấu" để thoát cực tiểu cục bộ)*

![Genetic Algorithms](https://github.com/user-attachments/assets/20d790d0-7812-462e-8019-448107f5900f)
**Genetic Algorithms (GA)**
*(Tiến hóa một quần thể các chuỗi nước đi)*

![Beam Search](https://github.com/user-attachments/assets/c7a0af09-12e4-4de8-a0cf-c7cabe1a3644)
**Beam Search**
*(Giữ lại một số trạng thái "hứa hẹn" nhất ở mỗi bước)*

#### 2.3.3. Hình ảnh so sánh hiệu suất của các thuật toán:
(Hiệu suất lý thuyết chung)
* **Hill Climbing:** Nhanh, đơn giản nhưng dễ bị kẹt ở cực trị địa phương.
* **Simulated Annealing:** Có khả năng tìm cực trị toàn cục tốt hơn, phụ thuộc "lịch trình làm nguội".
* **Genetic Algorithms:** Mạnh mẽ cho không gian phức tạp, có thể tìm giải pháp gần tối ưu, nhưng cần nhiều đánh giá.
* **Beam Search:** Cân bằng giữa Greedy và BFS, hiệu suất phụ thuộc "độ rộng chùm tia" (beam width).

![So sánh hiệu suất Nhóm 3](https://github.com/user-attachments/assets/ec7bc478-a931-4c55-b00e-d92f722dc74d)
*Test case: [[1, 2, 3], [4, 0, 5], [6, 7, 8]] (Trạng thái đích)*
*Ghi chú về hiệu suất thực tế trên 8-puzzle (tham khảo tệp log kết quả `8_puzzle_test_log.txt`):*
* Các biến thể Hill Climbing: Simple/Stochastic HC có thể bị kẹt ở các trạng thái không phải đích; Steepest HC ổn định hơn nhưng vẫn có thể bị kẹt.
* Simulated Annealing: Có thể tìm được đích nếu các tham số (nhiệt độ, tốc độ giảm nhiệt) được điều chỉnh tốt, nhưng quá trình khám phá thường dài và không đảm bảo tối ưu về số bước.
* Genetic Algorithm: Có thể tìm được lời giải tối ưu sau một số thế hệ, nhưng thời gian chạy có thể đáng kể và phụ thuộc vào các tham số như kích thước quần thể, tỷ lệ đột biến.
* Beam Search: Với `BEAM_WIDTH` hợp lý, có thể tìm ra lời giải tối ưu hiệu quả và nhanh chóng, đôi khi nhanh hơn A\* cho một số trường hợp.

#### 2.3.4. Nhận xét về hiệu suất trên trò chơi 8 ô chữ:
* **Hill Climbing:** Rất nhanh nhưng dễ bị kẹt ở cực trị địa phương (trạng thái không phải đích nhưng không có láng giềng nào tốt hơn).
* **Simulated Annealing (SA):** Có khả năng thoát khỏi cực trị địa phương, nhưng đường đi khám phá thường dài và không đảm bảo tìm được lời giải tối ưu về số bước.
* **Genetic Algorithm (GA):** Có thể tìm ra lời giải tối ưu nhưng cần thời gian và số thế hệ nhất định. Việc biểu diễn "cá thể" là một chuỗi nước đi là một cách tiếp cận thú vị.
* **Beam Search:** Với độ rộng chùm tia phù hợp, hoạt động tốt và nhanh chóng, có thể là một sự thay thế tốt cho A\* khi bộ nhớ là một vấn đề.

---

### 2.4. Nhóm 4: Tìm Kiếm Trong Môi Trường Phức Tạp

* **Tóm tắt nhóm:** Các thuật toán và phương pháp tiếp cận cho các môi trường có tính đối kháng, thông tin không đầy đủ (partially observable), hoặc động/không xác định. Dự án này minh họa một số khái niệm qua các kịch bản riêng trong `group4_algo.py`.
* **Các thuật toán/khái niệm chính (minh họa bằng các ví dụ riêng):** Tree Search AND-OR, Partially Observable environments, Unknown or Dynamic environments (minh họa qua Non-Observable Auto Random).

#### 2.4.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần (chung):**
    * **AND-OR Tree Search:** Phân rã bài toán thành các bài toán con (subproblems). Trạng thái có thể bao gồm trạng thái hiện tại của bài toán và trạng thái của các bài toán con.
    * **Partially Observable:** Agent không biết chắc chắn trạng thái hiện tại của môi trường, thay vào đó duy trì một **trạng thái niềm tin (belief state)** – một tập hợp các trạng thái vật lý có thể.
    * **Unknown/Dynamic Environment:** Agent cần học mô hình môi trường hoặc lập kế hoạch lại khi có thông tin mới.
* **Solution (chung):**
    * **AND-OR Tree Search:** Một cây giải pháp (AND-graph) thể hiện cách giải quyết tất cả các bài toán con cần thiết.
    * **Partially Observable:** Một chính sách (policy) ánh xạ từ belief state sang hành động.
    * **Unknown/Dynamic Environments:** Kế hoạch có điều kiện hoặc chính sách thích ứng.

* **Áp dụng trong `group4_algo.py`:**
    * **AND-OR Search Demo (Minh họa bằng 8-puzzle đặt từng ô):**
        * **Trạng thái:** Cấu hình bàn cờ 3x3 hiện tại và một tập hợp các ô đã được "cố định" (`fixed_tiles`) vào đúng vị trí. Mục tiêu con là đặt một ô chưa cố định vào đúng vị trí của nó.
        * **Solution (cho mỗi bước):** Một đường đi ngắn (tìm bằng A\* trong `a_star_solve_subgoal`) để di chuyển ô mục tiêu con đến vị trí đúng mà không làm xáo trộn các ô đã cố định. "Solution" tổng thể là việc hoàn thành tất cả các mục tiêu con.
    * **Partially Observable (Minh họa 8-puzzle với hàng đầu cố định, các hàng sau ẩn):**
        * **Trạng thái (Belief State):** Agent quản lý một tập hợp các cấu hình bàn cờ 3x3 (`current_belief_set`), mỗi cấu hình đều có hàng đầu là `[1,2,3]` và các hàng sau có thể khác nhau.
        * **Hành động:** Áp dụng một hành động (UP, DOWN, LEFT, RIGHT cho ô trống) cho *tất cả* các trạng thái trong belief set (nếu hợp lệ và không vi phạm ràng buộc hàng đầu).
        * **Solution:** Đạt được khi *một trong số* các trạng thái trong belief set trở thành trạng thái đích `[[1,2,3],[4,5,6],[7,8,0]]`.
    * **Non-Observable Auto Random (Minh họa với nhiều bàn cờ chạy song song):**
        * **Trạng thái:** Một tập hợp các bài toán 8-puzzle độc lập (`puzzles`), mỗi bài toán là một cấu hình bàn cờ 3x3.
        * **Hành động:** Một hành động ngẫu nhiên (UP, DOWN, LEFT, RIGHT cho ô trống) được áp dụng đồng thời cho *tất cả* các bài toán.
        * **Solution:** Đạt được khi *một trong số* các bài toán độc lập đó đạt được trạng thái đích chuẩn.

#### 2.4.2. Hình ảnh GIF của từng thuật toán/khái niệm khi áp dụng lên trò chơi:
![AND-OR Tree Search](https://github.com/user-attachments/assets/cf2fa2c9-db2b-4482-8b52-6d2387001bd4)
* **AND-OR Tree Search (Demo giải 8-puzzle theo từng mục tiêu con)**

![Partially Observable Environment](https://github.com/user-attachments/assets/98117603-0325-4257-9003-f73205ab9a26)
* **Partially Observable (Demo với belief states và hàng đầu cố định)**

![Unknown/Dynamic Environment](https://github.com/user-attachments/assets/250e3b65-9e2b-4d23-bbb9-591c8d20e806)
* **Unknown/Dynamic Environment (Demo Non-Observable với nhiều puzzle chạy song song)**

#### 2.4.3. Hình ảnh so sánh hiệu suất của các thuật toán/kịch bản:
* **AND-OR Trees:** Hiệu quả phụ thuộc vào cách phân rã bài toán và giải các bài toán con. Trong demo, A\* được dùng cho bài toán con.
* **Partially Observable:** Độ phức tạp tăng lên do phải quản lý không gian trạng thái niềm tin. Demo sử dụng hành động ngẫu nhiên.
* **Unknown/Dynamic Environments (Non-Observable Demo):** Demo cho thấy việc áp dụng hành động đồng thời, hiệu suất phụ thuộc vào may mắn và số lượng puzzle.

#### 2.4.4. Nhận xét về hiệu suất và ứng dụng (bao gồm trò chơi 8 ô chữ):
* Các thuật toán và khái niệm này được minh họa trên các kịch bản 8-puzzle đã được điều chỉnh trong `group4_algo.py`.
* **AND-OR Tree Search Demo:** Cách tiếp cận chia để trị có thể hiệu quả nếu các bài toán con dễ giải. Việc sử dụng A\* cho từng subgoal là hợp lý.
* **Partially Observable Demo:** Cho thấy thách thức khi thông tin không đầy đủ. Việc giải quyết thường phức tạp hơn nhiều so với môi trường quan sát đầy đủ.
* **Non-Observable Auto Random Demo:** Là một minh họa đơn giản về việc xử lý nhiều khả năng song song, không phải là một thuật toán tìm kiếm tối ưu cho môi trường không quan sát.

---

### 2.5. Nhóm 5: Tìm Kiếm Trong Môi Trường Có Ràng Buộc (Constraint Satisfaction Problems - CSPs)

* **Tóm tắt nhóm:** Tìm các giải pháp (phép gán giá trị cho biến) thỏa mãn một tập hợp các ràng buộc.
* **Các thuật toán chính (minh họa trong `group5_algo.py`):** Backtracking Search, Forward Checking (như một cải tiến của Backtracking), Min-Conflicts. AC-3 được dùng để hỗ trợ tạo bàn cờ.

#### 2.5.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần (chung cho CSP):**
    * **Tập biến (Variables):** Các đối tượng cần được gán giá trị.
    * **Tập miền giá trị (Domains):** Tập các giá trị có thể cho mỗi biến.
    * **Tập ràng buộc (Constraints):** Các điều kiện mà các phép gán giá trị phải thỏa mãn.
* **Solution (chung cho CSP):** Một phép gán giá trị đầy đủ và nhất quán cho tất cả các biến (tức là tất cả các ràng buộc đều được thỏa mãn).

* **Áp dụng trong `group5_algo.py` (tìm đường đi trong 8-puzzle với ràng buộc phụ):**
    * **Trạng thái (State):** Một cấu hình bàn cờ 3x3.
    * **Hành động & Hàm chuyển tiếp:** Di chuyển ô trống, nhưng có thể bị hạn chế bởi các ràng buộc bổ sung. Ví dụ, trong `Backtracking`, có ràng buộc mềm về việc giữ ô 3 và ô 6 kề nhau nếu chúng đã kề nhau.
    * **Solution:**
        * Với Backtracking và Forward Checking: Một danh sách các trạng thái (đường đi) từ trạng thái ban đầu đến trạng thái mục tiêu, sao cho tất cả các bước đi đều hợp lệ và (nếu có) thỏa mãn các ràng buộc động trong quá trình tìm kiếm.
        * Với Min-Conflicts: Một danh sách các trạng thái được duyệt qua để đến trạng thái cuối cùng. Trạng thái cuối cùng này là trạng thái có số "xung đột" (đánh giá bằng `Manhattan_Heuristic`) thấp nhất mà thuật toán tìm được sau một số lần lặp, hy vọng đó là trạng thái đích.
        * Với `AC3_Generate_Board`: "Solution" của hàm này là một trạng thái bàn cờ 3x3 hợp lệ, có thể giải được và thỏa mãn ràng buộc ô 3 và ô 6 phải kề nhau.

#### 2.5.2. Hình ảnh GIF của từng thuật toán khi áp dụng lên bài toán:
![Backtracking Search](https://github.com/user-attachments/assets/6d5d64f6-ae37-429d-8d18-a4debdae6ab0)
* **Backtracking Search (áp dụng cho 8-puzzle với ràng buộc phụ)**

![Forward Checking](https://github.com/user-attachments/assets/e0c03cc8-db6f-44c9-90cc-af3e5d739151)
* **Forward Checking (tương tự Backtracking, kiểm tra ràng buộc sớm hơn)**

![AC-3](https://github.com/user-attachments/assets/55463b30-d3eb-4b02-bb01-2590030b1d15)
* **AC-3 (Dùng để tạo bàn cờ 8-puzzle thỏa mãn ràng buộc 3&6 kề nhau)**

![Min-Conflicts](https://github.com/user-attachments/assets/20d3dc93-e43b-46af-bad0-91f713838055)
* **Min-Conflicts (Tìm trạng thái 8-puzzle đích bằng cách giảm xung đột)**

#### 2.5.3. Hình ảnh so sánh hiệu suất của các thuật toán:
* **Backtracking Search:** Cơ bản, có thể chậm nếu không có heuristic hoặc kỹ thuật cắt tỉa tốt.
* **Forward Checking & AC-3 (như kỹ thuật preprocessing/propagation):** Có thể cải thiện đáng kể hiệu suất của Backtracking bằng cách phát hiện xung đột sớm. Trong dự án, AC-3 dùng để tạo bàn cờ.
* **Min-Conflicts:** Thuật toán tìm kiếm cục bộ hiệu quả cho một số CSPs, nhưng có thể bị kẹt ở cực tiểu cục bộ.
* Hiệu suất rất phụ thuộc vào cấu trúc của bài toán và các ràng buộc cụ thể.

#### 2.5.4. Nhận xét về hiệu suất và ứng dụng (bao gồm trò chơi 8 ô chữ):
* Trong `group5_algo.py`, Backtracking và Forward Checking được áp dụng để tìm *đường đi* cho 8-puzzle, với một ràng buộc tùy chỉnh (ưu tiên giữ 3&6 kề nhau). Hiệu suất phụ thuộc vào độ sâu tìm kiếm và tính chặt của ràng buộc.
* `AC3_Generate_Board` là một ứng dụng thú vị của ý tưởng CSP để tạo ra các trường hợp thử nghiệm có đặc tính mong muốn.
* Min-Conflicts được dùng như một thuật toán tìm kiếm cục bộ để cố gắng đạt đến trạng thái đích của 8-puzzle bằng cách giảm thiểu heuristic Manhattan. Nó không đảm bảo tìm ra đường đi hoặc giải pháp tối ưu.

---

### 2.6. Nhóm 6: Học Tăng Cường (Reinforcement Learning - RL)

* **Tóm tắt nhóm:** Agent học cách hành động trong một môi trường để tối đa hóa một dạng phần thưởng tích lũy.
* **Thuật toán ví dụ (được triển khai cho 8-puzzle trong `main.py`):** Q-Learning.

#### 2.6.1. Thành phần chính của bài toán tìm kiếm và solution:
* **Thành phần (chung cho RL):** Agent, Môi trường, Trạng thái (State), Hành động (Action), Phần thưởng (Reward), Chính sách (Policy $\pi$).
* **Solution (chung cho RL):** Tìm ra một **chính sách tối ưu ($\pi\*$)** – một quy tắc chỉ dẫn agent nên chọn hành động nào ở mỗi trạng thái để tối đa hóa tổng phần thưởng kỳ vọng trong tương lai.

* **Áp dụng cho 8-puzzle với Q-Learning (trong `main.py`):**
    * **Agent:** Thực thể giải puzzle.
    * **Môi trường:** Trò chơi 8-puzzle và các quy tắc di chuyển ô trống.
    * **Trạng thái (State):** Một cấu hình bàn cờ 3x3, thường được biểu diễn dưới dạng tuple các tuple để có thể dùng làm key trong Q-table.
    * **Hành động (Action):** Di chuyển ô trống lên, xuống, trái, hoặc phải (được mã hóa thành các chỉ số 0, 1, 2, 3).
    * **Phần thưởng (Reward):**
        * `QL_REWARD_GOAL` (ví dụ: +100) khi đạt trạng thái đích.
        * `QL_REWARD_MOVE` (ví dụ: -1) cho mỗi bước đi không dẫn đến đích.
        * `QL_REWARD_PREVIOUS` (ví dụ: -10, không được sử dụng trong logic Q-Learning hiện tại nhưng có trong hằng số) có thể dùng để phạt việc quay lại trạng thái đã thăm.
    * **Solution:** Thuật toán Q-Learning học một **Q-table**, trong đó `Q(s, a)` lưu trữ giá trị kỳ vọng của việc thực hiện hành động `a` từ trạng thái `s`. Chính sách tối ưu được suy ra từ Q-table bằng cách chọn hành động có Q-value cao nhất ở mỗi trạng thái. "Solution" được hiển thị là đường đi (danh sách các trạng thái) được tạo ra bằng cách đi theo chính sách này từ trạng thái ban đầu đến đích.

#### 2.6.2. Hình ảnh GIF của thuật toán (ví dụ Q-Learning):
![qlearning](https://github.com/user-attachments/assets/284098bd-a219-45c5-a7bc-f6b5a93f18a7)
*(Minh họa agent học trong môi trường)*

#### 2.6.3. Hình ảnh so sánh hiệu suất của các thuật toán (trong RL nói chung):
* Đánh giá qua: Tốc độ hội tụ đến chính sách tối ưu, chất lượng của chính sách cuối cùng, Sample Efficiency (số lượng tương tác với môi trường cần thiết).
* **Q-Learning:** Là thuật toán model-free, off-policy. Hiệu suất phụ thuộc nhiều vào các tham số như tốc độ học (alpha), hệ số chiết khấu (gamma), chiến lược khám phá (epsilon-greedy và tốc độ giảm epsilon).
*Ghi chú về hiệu suất thực tế của Q-Learning trên 8-puzzle (tham khảo `main.py` và có thể ghi log nếu cần):*
* Q-Learning cần một số lượng đáng kể các "episodes" (lượt chơi thử) để học được một Q-table tốt, đặc biệt nếu bắt đầu từ một Q-table rỗng.
* Việc lưu và tải Q-table (`q_table_goal_{hash}_qstudy.pkl`) giúp tiết kiệm thời gian training cho các lần chạy sau.
* Đường đi được tái tạo từ Q-table đã học có thể không phải lúc nào cũng là đường đi ngắn nhất tuyệt đối (như A\* tìm ra) nếu Q-table chưa hội tụ hoàn toàn hoặc do tính ngẫu nhiên trong quá trình học và chọn hành động khi có nhiều Q-value bằng nhau.

#### 2.6.4. Nhận xét về hiệu suất trên trò chơi 8 ô chữ:
* 8-puzzle có thể mô hình hóa thành bài toán RL, và Q-Learning là một cách tiếp cận để giải quyết nó.
* Q-Learning cố gắng học hàm Q-value để suy ra chính sách tối ưu. Quá trình học có thể tốn thời gian (nhiều episodes).
* Sau khi học, việc tái tạo đường đi từ Q-table thường nhanh. Tuy nhiên, chất lượng của đường đi (tính tối ưu về số bước) phụ thuộc vào mức độ hội tụ của Q-table.
* So với các thuật toán tìm kiếm có thông tin như A\* (vốn được thiết kế để tìm đường đi tối ưu cho một lượt giải cụ thể), Q-Learning có thể không hiệu quả bằng cho bài toán 8-puzzle tĩnh nếu chỉ xét một lần giải. Sức mạnh của RL thường thể hiện rõ hơn trong các môi trường động, không chắc chắn, hoặc khi mô hình của môi trường không được biết trước.

---

## 3. Kết luận

* **Hệ thống hóa kiến thức:** Dự án đã tổng hợp và làm rõ các nhóm thuật toán tìm kiếm AI chính, từ cơ bản đến nâng cao.
* **Minh họa thực tiễn:** Triển khai và so sánh hiệu suất các thuật toán tìm kiếm trạng thái (Nhóm 1, 2, 3, 6) trên bài toán 8-puzzle, đồng thời minh họa các khái niệm tìm kiếm nâng cao (Nhóm 4, 5) qua các ví dụ và kịch bản phù hợp trên 8-puzzle hoặc các bài toán liên quan.
* **Cung cấp tài liệu tham khảo:** Tạo ra một nguồn tài liệu học tập về các thuật toán tìm kiếm, làm rõ cách mô hình hóa bài toán, biểu diễn trạng thái, định nghĩa giải pháp, cũng như ưu nhược điểm và tính phù hợp của từng nhóm cho các loại bài toán khác nhau.
* **Định hướng phát triển:** Đặt nền tảng cho việc khám phá sâu hơn các thuật toán, các kỹ thuật heuristic tiên tiến, và ứng dụng trong các lĩnh vực phức tạp hơn.
