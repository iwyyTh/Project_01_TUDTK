from manim import *
import numpy as np
import sys
import os

# Cấu hình path để import các module cùng cấp
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from diagonalization import faddeev_leverrier, solve_quadratic
from decomposition import svd_decomposition, _mat_mul, _transpose

# ==============================================================
# CẤU HÌNH HỆ THỐNG & MÀU SẮC
# ==============================================================
config.tex_compiler = "pdflatex"

class SVD_Visualized(ThreeDScene):

    def construct(self):
        # Thiết lập Palette màu sắc nhất quán
        self.colors = {
            "A": WHITE,
            "U": BLUE_C,
            "SIGMA": YELLOW_C,
            "VT": RED_C,
            "DIM": GRAY_B,
            "HL": ORANGE, # Highlight
            "BG": "#1a1a1a"
        }

        # LUỒNG KỊCH BẢN
        self.the_hook_scene()
        self.the_title_scene()
        self.the_problem_statement()
        self.the_svd_revelation()
        self.the_component_breakdown()
        self.the_component_detail()

        # Giả sử ma trận A đầu vào của bạn
        self.a_data = np.array([
            [3, 2], [2, 3], [2, -2]
        ])
        self.colors = {"A": BLUE, "U": GREEN, "SIGMA": YELLOW, "VT": RED, "HL": ORANGE}
        
        self.step1_compute_ata()
        self.step2_eigenvalues()
        self.step3_eigenvectors()
        self.step4_compute_sigma()
        self.step5_compute_u()
        self.step6_final_svd()

        self.setup_geometric_space()
        self.step_1_rotate_v_transpose()
        self.step_2_scale_sigma_3d()
        self.step_3_rotate_u_3d()
        self.back_to_2d()

        self.section1_intro()
        self.section2_build_ATA()
        self.section3_diagonalization()
        self.section4_bridge_svd()

        self.section5_svd_application()

        self.the_outro_scene()


    def clear_scene(self):
        if len(self.mobjects) > 0:
            self.play(FadeOut(*self.mobjects))    


    # ----------------------------------------------------------
    # 1. THE HOOK: SỰ HỖN LOẠN CỦA DỮ LIỆU
    # ----------------------------------------------------------
    def the_hook_scene(self):
        # 1. KHỞI TẠO DỮ LIỆU
        np.random.seed(10)
        num_dots = 500
        
        # Đám mây hỗn loạn ban đầu
        chaos_pos = [
            [np.random.uniform(-5, 5), np.random.uniform(-3, 2.5), 0]
            for _ in range(num_dots)
        ]
        
        # Cấu trúc cốt lõi (Đường thẳng)
        struct_pos = [
            [x, 0.4 * x + np.random.normal(0, 0.15), 0]
            for x in np.linspace(-4, 4, num_dots)
        ]
        
        dots = VGroup(*[
            Dot(point=chaos_pos[i], radius=0.03, color=GRAY_C, fill_opacity=0.5)
            for i in range(num_dots)
        ])

        self.play(FadeIn(dots))
        
        # Wiggle effect (Rung lắc)
        def wiggle_dots(mobs, dt):
            for mob in mobs:
                mob.shift(np.random.uniform(-0.02, 0.02, 3))

        dots.add_updater(wiggle_dots)
        self.wait(1.5)
        dots.remove_updater(wiggle_dots)

        # GIAI ĐOẠN 2: Alignment (Laser Scan)
        scan_line = Line(UP * 3.5, DOWN * 3.5, color=YELLOW).set_stroke(width=5)
        scan_line.set_glow_factor(1)
        scan_line.move_to(LEFT * 6)

        # Logic quét laser đến đâu đổi dot đến đó
        def update_dots(mobs):
            line_x = scan_line.get_x()
            for i, dot in enumerate(mobs):
                if dot.get_x() < line_x:
                    dot.move_to(struct_pos[i])
                    dot.set_color(BLUE_B)
                    dot.set_fill(opacity=0.8)
        
        dots.add_updater(update_dots)
        self.play(
            scan_line.animate.move_to(RIGHT * 6),
            run_time=3.5,
            rate_func=linear
        )
        dots.remove_updater(update_dots)
        self.play(FadeOut(scan_line))

        # GIAI ĐOẠN 3: Reveal
        svd_text = Text("SVD", font_size=120, weight=BOLD).set_color_by_gradient(BLUE, YELLOW)
        sub_text = Text("Singular Value Decomposition", font_size=32, color=GRAY_A).next_to(svd_text, DOWN)
        
        self.play(
            ReplacementTransform(dots, svd_text),
            run_time=2
        )
        self.play(Write(sub_text), Flash(svd_text, color=YELLOW, num_lines=20))
        self.wait(2)

        # Cleanup để sang cảnh tiếp theo
        self.play(FadeOut(svd_text), FadeOut(sub_text))


    # ----------------------------------------------------------
    # 2. TITLE SCENE
    # ----------------------------------------------------------
    
    def the_title_scene(self):
        # Đổi thành "Giải Mã Cấu Trúc" hoặc "Bản Chất Ma Trận"
        title = Text("Giới Thiệu", font_size=48, color=WHITE, weight=BOLD)
        
        # Làm cái gạch chân trông xịn hơn bằng cách dùng Gradient
        underline = Line(LEFT, RIGHT).set_width(title.get_width() * 1.2)
        underline.set_color_by_gradient(BLUE, YELLOW)
        underline.next_to(title, DOWN, buff=0.2)
        
        self.intro_title_group = VGroup(title, underline)
        
        # Hiệu ứng Write kết hợp với hiệu ứng khối (Glow) nhẹ
        self.play(
            Write(title), 
            Create(underline),
            run_time=1.5
        )
        self.wait(1)
        
        # Thu nhỏ dạt lên góc để nhường sân khấu cho công thức sắp xuất hiện
        self.play(
            self.intro_title_group.animate.to_edge(UP, buff=0.4).scale(0.6),
            run_time=1
        )

    # ----------------------------------------------------------
    # 3. ĐẶT VẤN ĐỀ: GIỚI HẠN CỦA CHÉO HÓA 
    # ----------------------------------------------------------

    def the_problem_statement(self):
        # Ma trận A (Dịch sang trái một chút để lấy chỗ)
        matrix_data = [[3, 2], [2, 3], [2, -2]]
        matrix_a = Matrix(matrix_data, left_bracket="[", right_bracket="]").set_color(self.colors["A"])
        label_a = MathTex("A = ").next_to(matrix_a, LEFT)
        desc_a = Text("Ma trận m x n (Hình chữ nhật)", font_size=20, color=GRAY_B).next_to(matrix_a, DOWN)
        
        a_group = VGroup(label_a, matrix_a, desc_a).to_edge(LEFT, buff=1.0) # Đẩy sát lề trái

        self.play(DrawBorderThenFill(matrix_a), Write(label_a))
        self.play(FadeIn(desc_a, shift=UP))
        self.wait(1)

        # Cụm thông báo lỗi (Dịch sang phải)
        diag_formula = MathTex(
            "A", "=", "P", "D", "P^{-1}",
            tex_to_color_map={"P": BLUE, "D": YELLOW, "P^{-1}": RED}
        ).shift(RIGHT * 2.5 + UP * 0.5) # Dịch sang phải 2.5 đơn vị
        
        diag_warn = Text(
            "Lỗi: Chỉ áp dụng cho ma trận VUÔNG", 
            font_size=22, color=RED_C
        ).next_to(diag_formula, DOWN, buff=0.4) # Tăng buff để giãn cách dòng

        cross = Cross(diag_formula, stroke_width=2, color=RED)

        # Hiệu ứng
        self.play(Write(diag_formula))
        self.wait(0.5)
        # Gạch chéo và hiện cảnh báo cùng lúc, dời nhẹ diag_warn lên để thấy rõ
        self.play(
            Create(cross), 
            FadeIn(diag_warn, shift=RIGHT * 0.5) # Hiệu ứng hiện ra dịch từ trái sang phải
        )
        self.wait(2)

        # Dọn dẹp màn hình trước khi sang phần tiếp theo
        self.play(FadeOut(diag_formula, cross, diag_warn, a_group))


    # ----------------------------------------------------------
    # 4. CÔNG THỨC SVD TỔNG QUÁT (ĐÃ FIX KEYERROR)
    # ----------------------------------------------------------
    def the_svd_revelation(self):
        self.svd_formula = MathTex("A", "=", "U", "\\Sigma", "V^T")
        self.svd_formula.set_color_by_tex_to_color_map({
            "A": self.colors["A"],
            "U": self.colors["U"],
            "\\Sigma": self.colors["SIGMA"],
            "V^T": self.colors["VT"]
        })

        self.play(Write(self.svd_formula))
        self.wait(1)
        self.play(self.svd_formula.animate.to_edge(UP, buff=1.2).scale(0.5))


    # ----------------------------------------------------------
    # 5. CHI TIẾT CẤU TRÚC (ĐÃ THÊM PHẦN DỌN DẸP)
    # ----------------------------------------------------------
    def the_component_breakdown(self):
        unit = 0.8
        
        def create_block(h, w, color, label):
            rect = Rectangle(
                height=h*unit, width=w*unit, 
                color=color, fill_color=color, fill_opacity=0.2
            )
            lbl = MathTex(label, color=color).move_to(rect.get_center())
            return VGroup(rect, lbl)

        block_a = create_block(3, 2, self.colors["A"], "A")
        block_u = create_block(3, 3, self.colors["U"], "U")
        block_s = create_block(3, 2, self.colors["SIGMA"], "\\Sigma")
        block_v = create_block(2, 2, self.colors["VT"], "V^T")

        # Lưu vào Group để dễ quản lý và xóa
        layout_elements = VGroup(
            block_a, MathTex("="), block_u, MathTex("\\cdot"), 
            block_s, MathTex("\\cdot"), block_v
        ).arrange(RIGHT, buff=0.4).shift(UP * 0.5) # Dịch lên trên một chút để lấy chỗ cho note

        dim_labels = VGroup(
            self.get_dim("m \\times n", block_a),
            self.get_dim("m \\times m", block_u),
            self.get_dim("m \\times n", block_s),
            self.get_dim("n \\times n", block_v)
        )

        self.play(FadeIn(block_a), Write(dim_labels[0]))
        self.wait(1)

        self.play(
            ReplacementTransform(block_a.copy(), block_u),
            ReplacementTransform(block_a.copy(), block_s),
            ReplacementTransform(block_a.copy(), block_v),
            Write(layout_elements[1]), Write(layout_elements[3]), Write(layout_elements[5]),
            run_time=2
        )
        self.play(Write(dim_labels[1:]))
        self.wait(2)

        notes = VGroup(
            Text("• U : Ma trận trực giao (Left Singular Vectors)", font_size=18, color=self.colors["U"]),
            Text("• Σ : Ma trận giá trị suy biến (Singular Values)", font_size=18, color=self.colors["SIGMA"]),
            Text("• Vᵀ: Ma trận trực giao (Right Singular Vectors)", font_size=18, color=self.colors["VT"])
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(DOWN, buff=0.7)

        self.play(LaggedStartMap(FadeIn, notes, shift=RIGHT, lag_ratio=0.3))
        self.wait(4)

        # === ĐOẠN QUAN TRỌNG: DỌN DẸP MÀN HÌNH ĐỂ KHÔNG BỊ ĐÈ ===
        self.play(
            FadeOut(layout_elements),
            FadeOut(dim_labels),
            FadeOut(notes),
            run_time=1.5
        )
        self.wait(0.5)

    def get_dim(self, text, target):
        return MathTex(text, font_size=20, color=self.colors["DIM"]).next_to(target, DOWN, buff=0.2)
    

    # ----------------------------------------------------------
    # 6. CHI TIẾT TỪNG THÀNH PHẦN (BẢN FIX: CÓ CÔNG THỨC THAM CHIẾU MỚI)
    # ----------------------------------------------------------
    def the_component_detail(self):
        # 1. Xóa sạch mọi thứ từ các phần trước để làm "tờ giấy trắng"
        self.play(FadeOut(*self.mobjects)) 
        self.wait(0.2)

        # 2. Hiện tiêu đề PHÂN TÍCH CHI TIẾT
        detail_title = Text("PHÂN TÍCH CHI TIẾT", font_size=32, color=self.colors["HL"])
        detail_title.to_edge(UP, buff=0.3)
        self.play(Write(detail_title))

        # 3. Tạo một công thức SVD mới nằm ngay dưới tiêu đề để tham chiếu
        # Công thức này tạo mới hoàn toàn, không dùng lại biến cũ
        ref_formula = MathTex(
            "A", "=", "U", "\\Sigma", "V^T",
            font_size=36
        ).next_to(detail_title, DOWN, buff=0.3)
        
        ref_formula.set_color_by_tex_to_color_map({
            "A": self.colors["A"],
            "U": self.colors["U"],
            "\\Sigma": self.colors["SIGMA"],
            "V^T": self.colors["VT"]
        })
        
        # Vẽ một cái khung bao quanh công thức tham chiếu cho đẹp
        ref_box = SurroundingRectangle(ref_formula, color=GRAY_D, buff=0.2, stroke_width=1)
        
        self.play(Write(ref_formula), Create(ref_box))
        self.wait(0.5)

        # 4. Hiển thị 3 cột thông tin chi tiết (Dời xuống thấp một chút để không đè)
        # Cột U
        u_group = VGroup(
            MathTex("U : m \\times m", color=self.colors["U"], font_size=26),
            Matrix([["u_1", "u_2", "\\dots", "u_m"]]).set_color(self.colors["U"]).scale(0.5),
            Text("• Ma trận trực giao", font_size=16),
            MathTex("U^T U = I", font_size=20, color=self.colors["U"])
        ).arrange(DOWN, buff=0.3).shift(LEFT * 4.5 + DOWN * 1)

        # Cột Sigma
        sigma_group = VGroup(
            MathTex("\\Sigma : m \\times n", color=self.colors["SIGMA"], font_size=26),
            Matrix([["\\sigma_1", "0", "\\dots"], ["0", "\\sigma_2", "\\dots"], ["0", "0", "0"]]).set_color(self.colors["SIGMA"]).scale(0.5),
            MathTex("\\sigma_1 \\ge \\sigma_2 \\ge \\dots \\ge 0", font_size=20, color=self.colors["SIGMA"])
        ).arrange(DOWN, buff=0.3).shift(DOWN * 1)

        # Cột V^T
        v_group = VGroup(
            MathTex("V^T : n \\times n", color=self.colors["VT"], font_size=26),
            Matrix([["v_1^T"], ["v_2^T"], ["\\vdots"]]).set_color(self.colors["VT"]).scale(0.5),
            Text("• Ma trận trực giao", font_size=16),
            MathTex("V^T V = I", font_size=20, color=self.colors["VT"])
        ).arrange(DOWN, buff=0.3).shift(RIGHT * 4.5 + DOWN * 1)

        # Diễn hoạt các cột hiện ra đồng thời
        self.play(
            LaggedStart(
                FadeIn(u_group, shift=UP, scale=0.9),     # Thêm scale nhẹ cho sinh động
                FadeIn(sigma_group, shift=UP, scale=0.9),
                FadeIn(v_group, shift=UP, scale=0.9),
                lag_ratio=0.5 # Tăng độ trễ giữa U, Sigma và V
            ),
            run_time=4, # Tổng thời gian xuất hiện là 4 giây (rất từ từ)
            rate_func=slow_into # Hiệu ứng vào chậm, giữa nhanh, kết thúc chậm
        )
        
        self.wait(4)

        # 5. Xóa sạch để kết thúc phân đoạn
        self.play(FadeOut(detail_title, ref_formula, ref_box, u_group, sigma_group, v_group))


    # ----------------------------------------------------------
    # 7. CHI TIẾT QUÁ TRÌNH PHÂN RÃ (CẬP NHẬT BƯỚC TÌM U, V)
    # ----------------------------------------------------------
    def step1_compute_ata(self):
        self.play(FadeOut(*self.mobjects))
        
        # Tiêu đề cố định ở trên
        title = Text("Bước 1: Tính ma trận hiệp phương sai AᵀA", font_size=32, color=self.colors["HL"]).to_edge(UP, buff=0.5)
        self.play(Write(title))

        # Khởi tạo dữ liệu
        A_mat = Matrix(self.a_data).scale(0.7)
        A_label = MathTex("A =").next_to(A_mat, LEFT)
        A_group = VGroup(A_label, A_mat)

        At = _transpose(self.a_data)
        At_mat = Matrix(np.round(At, 2)).scale(0.7)
        At_label = MathTex("A^T =").next_to(At_mat, LEFT)
        At_group = VGroup(At_label, At_mat)

        # Sắp xếp A và A^T để chuẩn bị nhân
        calculation_ui = VGroup(At_group, A_group).arrange(RIGHT, buff=1).center()
        
        self.play(FadeIn(calculation_ui, shift=UP))
        self.wait(1)

        # Tính toán kết quả
        ATA = _mat_mul(At, self.a_data)
        self.ATA = ATA
        
        res_mat = Matrix(np.round(ATA, 1)).set_color(YELLOW).scale(0.8)
        res_label = MathTex("A^TA =").next_to(res_mat, LEFT)
        res_group = VGroup(res_label, res_mat).center().shift(DOWN*0.5)

        self.play(
            calculation_ui.animate.scale(0.6).to_edge(UP, buff=1.5),
            FadeIn(res_group, shift=UP)
        )
        self.wait(2)

    def step2_eigenvalues(self):
        self.play(FadeOut(*self.mobjects))
        
        title = Text("Bước 2: Tìm Trị riêng (Eigenvalues)", font_size=32, color=self.colors["HL"]).to_edge(UP, buff=0.5)
        step_desc = Text("Giải phương trình đặc trưng: det(AᵀA - λI) = 0", font_size=20, color=GRAY_B).next_to(title, DOWN)
        self.play(Write(title), Write(step_desc))

        # Hiển thị lại ma trận ATA nhỏ gọn làm mốc
        # 1. Tạo ma trận và nhãn
        ata_brief = Matrix(np.round(self.ATA, 1)).scale(0.6)
        ata_lbl = MathTex("A^TA =").scale(0.8)
        
        # 2. Gom vào Group và sắp xếp thứ tự
        ata_side_group = VGroup(ata_lbl, ata_brief).arrange(RIGHT, buff=0.2)
        
        # 3. Đẩy vào góc trên bên trái (UL - Upper Left) 
        # Cách này an toàn hơn to_edge(LEFT) vì nó tính toán cả khoảng cách trên dưới
        ata_side_group.to_corner(UL, buff=1.5) 
        
        # Nếu vẫn thấy hơi sát mép, bạn có thể shift nhẹ lại
        # ata_side_group.shift(RIGHT * 0.5)

        self.play(FadeIn(ata_side_group))

        # Logic tính toán
        coeffs = faddeev_leverrier(self.ATA)
        eigenvalues = sorted([ev.real for ev in solve_quadratic(coeffs[1], coeffs[0])], reverse=True)
        self.eigenvalues = eigenvalues

        # Hiển thị phương trình bậc 2 (giả định từ hệ số)
        poly = MathTex("\\lambda^2 - ", f"{int(coeffs[1])}", "\\lambda + ", f"{int(coeffs[0])}", "= 0").shift(RIGHT*2)
        self.play(Write(poly))
        self.wait(1)

        result = MathTex(
            f"\\lambda_1 = {round(eigenvalues[0], 2)}", 
            ", \\quad ", 
            f"\\lambda_2 = {round(eigenvalues[1], 2)}"
        ).set_color(self.colors["SIGMA"]).next_to(poly, DOWN, buff=0.8)

        self.play(FadeIn(result, shift=UP))
        self.wait(3)

    def step3_eigenvectors(self):
        self.play(FadeOut(*self.mobjects))
        
        title = Text("Bước 3: Tìm Vector riêng (Eigenvectors)", font_size=32, color=self.colors["HL"]).to_edge(UP, buff=0.5)
        step_desc = MathTex("(A^TA - \\lambda I)\\mathbf{v} = \\mathbf{0}").scale(0.8).next_to(title, DOWN)
        self.play(Write(title), Write(step_desc))

        # Lấy dữ liệu SVD đã tính sẵn (V là vector riêng của ATA)
        U_val, S_val, Vt_val = svd_decomposition(self.a_data)
        V_val = _transpose(Vt_val)
        self.V = V_val
        self.Vt = Vt_val

        # Trình bày quá trình gom Vector
        v1 = Matrix([[round(V_val[0][0], 2)], [round(V_val[1][0], 2)]]).set_color(self.colors["VT"]).scale(0.7)
        v2 = Matrix([[round(V_val[0][1], 2)], [round(V_val[1][1], 2)]]).set_color(self.colors["VT"]).scale(0.7)
        
        v_group = VGroup(
            VGroup(MathTex("\\mathbf{v}_1="), v1).arrange(RIGHT),
            VGroup(MathTex("\\mathbf{v}_2="), v2).arrange(RIGHT)
        ).arrange(RIGHT, buff=1.5).center()

        self.play(LaggedStart(*[FadeIn(obj) for obj in v_group], lag_ratio=0.5))
        self.wait(1)

        # Gom thành ma trận V
        V_mat = Matrix(np.round(V_val, 2)).set_color(self.colors["VT"]).scale(0.8)
        V_lbl = MathTex("V = [\\mathbf{v}_1 \\; \\mathbf{v}_2] =").next_to(V_mat, LEFT)
        final_V_ui = VGroup(V_lbl, V_mat).center()

        self.play(ReplacementTransform(v_group, final_V_ui))
        self.wait(3)

    def step4_compute_sigma(self):
        self.play(FadeOut(*self.mobjects))
        
        title = Text("Bước 4: Tính Trị suy biến (Singular Values)", font_size=32, color=self.colors["HL"]).to_edge(UP, buff=0.5)
        formula = MathTex("\\sigma_i = \\sqrt{\\lambda_i}").next_to(title, DOWN, buff=0.5)
        self.play(Write(title), Write(formula))

        s1, s2 = np.sqrt(self.eigenvalues[0]), np.sqrt(self.eigenvalues[1])
        self.sigma = [s1, s2]

        res = MathTex(
            f"\\sigma_1 = \\sqrt{{{round(self.eigenvalues[0],1)}}} = {round(s1, 2)}",
            "\\\\",
            f"\\sigma_2 = \\sqrt{{{round(self.eigenvalues[1],1)}}} = {round(s2, 2)}"
        ).set_color(self.colors["SIGMA"]).shift(DOWN*0.5)

        self.play(Write(res))
        
        # Hiện ma trận Sigma (3x2)
        Sigma_mat = Matrix([[round(s1,2), 0], [0, round(s2,2)], [0, 0]]).scale(0.7).to_edge(RIGHT, buff=1.5)
        Sigma_lbl = MathTex("\\Sigma =").next_to(Sigma_mat, LEFT)
        
        self.play(FadeIn(VGroup(Sigma_lbl, Sigma_mat), shift=LEFT))
        self.wait(3)

    def step5_compute_u(self):
        self.play(FadeOut(*self.mobjects))
        
        title = Text("Bước 5: Tìm ma trận Vector kỳ dị trái U", font_size=32, color=self.colors["HL"]).to_edge(UP, buff=0.5)
        formula = MathTex("\\mathbf{u}_i = \\frac{1}{\\sigma_i} A \\mathbf{v}_i").next_to(title, DOWN, buff=0.3)
        self.play(Write(title), Write(formula))

        # Lấy U đã tính từ hàm svd
        U_val, _, _ = svd_decomposition(self.a_data)
        self.U = U_val

        U_mat = Matrix(np.round(U_val, 2)).set_color(self.colors["U"]).scale(0.7)
        U_lbl = MathTex("U = [\\mathbf{u}_1 \\; \\mathbf{u}_2 \\; \\mathbf{u}_3] =").next_to(U_mat, LEFT)
        U_ui = VGroup(U_lbl, U_mat).center().shift(DOWN*0.5)

        self.play(FadeIn(U_ui, shift=UP))
        self.wait(3)

    def step6_final_svd(self):
        self.play(FadeOut(*self.mobjects))
        
        title = Text("KẾT QUẢ PHÂN RÃ SVD", font_size=36, color=self.colors["HL"]).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Chuẩn bị 3 ma trận
        U_m = Matrix(np.round(self.U, 2)).set_color(self.colors["U"]).scale(0.5)
        
        Sigma_val = np.zeros((3,2))
        Sigma_val[0,0], Sigma_val[1,1] = self.sigma[0], self.sigma[1]
        S_m = Matrix(np.round(Sigma_val, 2)).set_color(self.colors["SIGMA"]).scale(0.5)
        
        Vt_m = Matrix(np.round(self.Vt, 2)).set_color(self.colors["VT"]).scale(0.5)

        # Nhãn tên ma trận ở trên đầu
        u_l = MathTex("U", color=self.colors["U"]).next_to(U_m, UP)
        s_l = MathTex("\\Sigma", color=self.colors["SIGMA"]).next_to(S_m, UP)
        vt_l = MathTex("V^T", color=self.colors["VT"]).next_to(Vt_m, UP)

        # Dấu bằng và nhân
        eq = MathTex("A =").scale(0.8)
        
        # Sắp xếp thủ công Group để đẹp
        final_group = VGroup(eq, U_m, S_m, Vt_m).arrange(RIGHT, buff=0.4).center()
        labels = VGroup(u_l, s_l, vt_l)
        
        # Căn chỉnh lại nhãn sau khi arrange
        u_l.next_to(U_m, UP)
        s_l.next_to(S_m, UP)
        vt_l.next_to(Vt_m, UP)

        self.play(
            FadeIn(final_group),
            FadeIn(labels)
        )
        
        summary = MathTex("A_{3 \\times 2} = U_{3 \\times 3} \\Sigma_{3 \\times 2} V^T_{2 \\times 2}").to_edge(DOWN, buff=1)
        self.play(Write(summary))
        self.wait(5)

        self.clear_scene()


    # ----------------------------------------------------------
    # 8. TRỰC QUAN HÓA HÌNH HỌC SVD (3x2 MATRIX)
    # Rotate → Scale → Rotate
    # ----------------------------------------------------------
    def setup_geometric_space(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

        # UI cố định
        self.main_title = Text(
            "Trực Quan Hóa SVD: Rotate → Scale → Rotate",
            font_size=28,
            color=self.colors["HL"]
        ).to_edge(UP, buff=0.3)

        self.border_frame = Rectangle(
            width=11.0,
            height=6.0,
            color=WHITE,
            stroke_width=1.5
        )

        self.add_fixed_in_frame_mobjects(
            self.main_title,
            self.border_frame
        )

        # 3D axes
        self.axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-4, 4, 1]
        )

        # Circle outline
        self.unit_circle = ParametricFunction(
            lambda t: np.array([
                np.cos(t),
                np.sin(t),
                0
            ]),
            t_range=[0, TAU],
            color=WHITE,
            stroke_width=3
        )

        # Circle fill (mờ)
        self.circle_fill = Surface(
            lambda u, v: np.array([
                u*np.cos(v),
                u*np.sin(v),
                0
            ]),
            u_range=[0,1],
            v_range=[0,TAU],
            fill_opacity=0.15,
            fill_color=BLUE,
            stroke_width=0
        )

        # vectors
        self.v1 = Vector([1,0,0], color=self.colors["VT"])
        self.v2 = Vector([0,1,0], color=self.colors["VT"])

        # labels
        self.v_labels = VGroup(
            MathTex("v_1"),
            MathTex("v_2")
        )

        self.v_labels[0].add_updater(
            lambda m: m.next_to(
                self.v1.get_end(),
                RIGHT,
                buff=0.1
            )
        )

        self.v_labels[1].add_updater(
            lambda m: m.next_to(
                self.v2.get_end(),
                UP,
                buff=0.1
            )
        )

        self.play(
            Write(self.main_title),
            Create(self.border_frame)
        )

        self.play(
            FadeIn(self.axes),
            FadeIn(self.circle_fill),
            Create(self.unit_circle)
        )

        self.play(
            GrowArrow(self.v1),
            GrowArrow(self.v2),
            Write(self.v_labels)
        )

        self.wait()

    def step_1_rotate_v_transpose(self):

        # Tạo label
        step_label = Text("Bước 1: Rotate (Vᵀ)", font_size=22, color=self.colors["VT"])
        
        # Căn chỉnh dựa trên border_frame
        step_label.align_to(self.border_frame, UP).align_to(self.border_frame, LEFT)
        step_label.shift(DOWN * 0.5 + RIGHT * 0.5) # Đẩy vào trong khung

        self.add_fixed_in_frame_mobjects(step_label)
        
        # Các logic tính toán xoay giữ nguyên...
        self.play(Write(step_label))

        # Vt từ đề bài
        VT = np.array([
            [0.71, 0.71],
            [-0.71, 0.71]
        ])

        angle = np.arctan2(VT[0,1], VT[0,0])

        rotation_group = VGroup(
            self.unit_circle,
            self.circle_fill,
            self.v1,
            self.v2
        )

        # Xoay camera sang phải 45°
        self.move_camera(
            theta = -65 * DEGREES,   # xoay phải thêm
            phi = 65 * DEGREES,
            run_time = 2
        )
        self.wait(0.5)

        # Rotate theo V^T
        self.play(
            Rotate(
                rotation_group,
                angle=angle,
                axis=OUT,
                about_point=ORIGIN
            ),
            run_time=2
        )

        

        

        self.wait()
        self.play(FadeOut(step_label))

    def step_2_scale_sigma_3d(self):

        step_label = Text("Bước 2: Scale (Σ)", font_size=22, color=self.colors["SIGMA"])
        
        # Căn chỉnh y hệt bước 1 để chữ xuất hiện đúng vị trí đó
        step_label.align_to(self.border_frame, UP).align_to(self.border_frame, LEFT)
        step_label.shift(DOWN * 0.5 + RIGHT * 0.5)

        self.add_fixed_in_frame_mobjects(step_label)
        
        # Các logic camera zoom và biến đổi giữ nguyên...
        self.play(Write(step_label))

        # Sigma từ đề bài
        S = np.array([5.0, 3.0])

        direction_v1 = self.v1.get_vector()
        direction_v1 /= np.linalg.norm(direction_v1)

        direction_v2 = self.v2.get_vector()
        direction_v2 /= np.linalg.norm(direction_v2)

        target_v1 = direction_v1 * S[0]
        target_v2 = direction_v2 * S[1]

        ellipse = ParametricFunction(
            lambda t:
            S[0]*np.cos(t)*direction_v1 +
            S[1]*np.sin(t)*direction_v2,
            t_range=[0, TAU],
            color=WHITE,
            stroke_width=3
        )

        ellipse_fill = Surface(
            lambda u, v:
            u*(S[0]*np.cos(v)*direction_v1 +
            S[1]*np.sin(v)*direction_v2),
            u_range=[0,1],
            v_range=[0,TAU],
            fill_color=YELLOW,
            fill_opacity=0.18,
            stroke_width=0
        )

        self.sigma_labels = VGroup(
            MathTex("\\sigma_1 = 5", color=YELLOW),
            MathTex("\\sigma_2 = 3", color=YELLOW)
        )

        self.sigma_labels[0].add_updater(
            lambda m: m.next_to(self.v1.get_end(), UP)
        )

        self.sigma_labels[1].add_updater(
            lambda m: m.next_to(self.v2.get_end(), RIGHT)
        )

        self.move_camera(
            zoom=0.65,
            run_time=3,
            added_anims=[
                Transform(self.unit_circle, ellipse),
                Transform(self.circle_fill, ellipse_fill),
                self.v1.animate.put_start_and_end_on(ORIGIN, target_v1),
                self.v2.animate.put_start_and_end_on(ORIGIN, target_v2),
                FadeIn(self.sigma_labels)
            ]
        )

        self.wait(2)
        self.play(FadeOut(step_label))

    def step_3_rotate_u_3d(self):
        """
        Bước 3: Rotate (U)
        Hiển thị vector đơn vị + vector kỳ dị trái
        """

        step_label = Text(
            "Bước 3: Rotate (U)",
            font_size=24,
            color=self.colors["U"]
        ).to_corner(UL, buff=1.5)

        # U từ đề bài
        U = np.array([
            [0.71, -0.24, 0.67],
            [0.71,  0.24,-0.67],
            [0.0,  -0.94,-0.33]
        ])

        # Fix left-handed matrix
        if np.linalg.det(U) < 0:
            U[:, -1] *= -1

        # --------------------------------------------------
        # Hiển thị Ma trận U
        # --------------------------------------------------

        u_mat = Matrix(np.round(U, 2))\
            .set_color(self.colors["U"])\
            .scale(0.5)

        u_label = Text(
            "Left Singular Vectors (U)", 
            font_size=16  # Giảm nhẹ font size để trông tinh tế hơn
        ).next_to(u_mat, UP, buff=0.2)

        u_ui_group = VGroup(u_label, u_mat)

        # CĂN CHỈNH CHIẾN THUẬT:
        # 1. Đưa group vào góc trên bên phải của CÁI KHUNG (không phải màn hình)
        u_ui_group.align_to(self.border_frame, UP).align_to(self.border_frame, RIGHT)
        
        # 2. Shift lùi vào trong khung một chút để không bị dính sát viền
        # Xuống dưới 0.4 (để tránh tiêu đề) và sang trái 0.4
        u_ui_group.shift(DOWN * 0.5 + LEFT * 0.5)

        self.add_fixed_in_frame_mobjects(
            step_label, 
            u_ui_group
        )

        # Căn chỉnh lại step_label tương tự để không đè khung trái
        step_label.align_to(self.border_frame, UP).align_to(self.border_frame, LEFT)
        step_label.shift(DOWN * 0.5 + RIGHT * 0.5)

        self.play(
            Write(step_label),
            FadeIn(u_ui_group, shift=LEFT)
        )

        # --------------------------------------------------
        # Vector kỳ dị trái (u1 u2 u3)
        # --------------------------------------------------

        u_vectors = VGroup(*[
            Vector(U[:, i], color=YELLOW)
            for i in range(3)
        ])

        u_labels = VGroup(*[
            MathTex(f"u_{i+1}", color=YELLOW)
            for i in range(3)
        ])

        for i in range(3):
            u_labels[i].add_updater(
                lambda m, i=i:
                m.next_to(
                    u_vectors[i].get_end(),
                    RIGHT
                )
            )

        self.play(
            LaggedStart(
                *[GrowArrow(v) for v in u_vectors],
                lag_ratio=0.3
            ),
            FadeIn(u_labels),
            run_time=2
        )

        self.wait(1)

        # --------------------------------------------------
        # Vector đơn vị ban đầu (Ox Oy Oz)
        # --------------------------------------------------

        unit_vectors = VGroup(
            Vector([1,0,0], color=YELLOW),
            Vector([0,1,0], color=YELLOW),
            Vector([0,0,1], color=YELLOW)
        )

        unit_vectors.set_opacity(0.35)

        unit_labels = VGroup(
            MathTex("e_1", color=YELLOW),
            MathTex("e_2", color=YELLOW),
            MathTex("e_3", color=YELLOW)
        )

        for i in range(3):
            unit_labels[i].add_updater(
                lambda m, i=i:
                m.next_to(
                    unit_vectors[i].get_end(),
                    RIGHT
                )
            )

        self.play(
            LaggedStart(
                *[GrowArrow(v) for v in unit_vectors],
                lag_ratio=0.2
            ),
            FadeIn(unit_labels),
            run_time=2
        )

        self.wait(1)

        # --------------------------------------------------
        # Rotate ellipse + vector đơn vị
        # --------------------------------------------------

        graph_group = VGroup(
            self.unit_circle,
            self.circle_fill,
            self.v1,
            self.v2,
            self.sigma_labels,
            unit_vectors
        )

        final_formula = MathTex(
            "A = U \\Sigma V^T",
            font_size=42,
            color=self.colors["HL"]
        ).to_edge(DOWN, buff=0.5)

        self.add_fixed_in_frame_mobjects(
            final_formula
        )

        # Rotation 3D chuẩn
        from scipy.spatial.transform import Rotation as R

        rotation = R.from_matrix(U)

        rotvec = rotation.as_rotvec()
        angle = np.linalg.norm(rotvec)
        axis = rotvec / angle

        self.move_camera(
            phi=75 * DEGREES,
            theta=-20 * DEGREES,
            run_time=3,
            added_anims=[
                Rotate(
                    graph_group,
                    angle=angle,
                    axis=axis,
                    about_point=ORIGIN
                ),
                Write(final_formula)
            ]
        )

        self.wait(1)

        # --------------------------------------------------
        # Dọn dẹp
        # --------------------------------------------------

        self.play(
            FadeOut(unit_vectors),
            FadeOut(unit_labels),
            u_ui_group.animate.set_opacity(0.3),
            run_time=1
        )

        # Camera cinematic

        self.begin_ambient_camera_rotation(
            rate=0.15
        )

        conclusion_text = Text(
            "Ellipse xoay theo Left Singular Vectors",
            font_size=20,
            color=self.colors["HL"]
        ).next_to(final_formula, UP, buff=0.3)

        self.add_fixed_in_frame_mobjects(
            conclusion_text
        )

        self.play(
            FadeIn(conclusion_text)
        )

        self.wait(5)

        self.stop_ambient_camera_rotation()

    def back_to_2d(self):
        self.clear_scene() # Xóa sạch mọi thứ đang hiện
        self.set_camera_orientation(phi=0, theta=-90*DEGREES) # Reset góc nhìn 2D


    # ----------------------------------------------------------
    # 9. CHÉO HÓA & LIÊN HỆ
    # ----------------------------------------------------------
    def section1_intro(self):

        title = Text(
            "Mối liên hệ giữa SVD và Chéo hóa",
            font_size=42,
            color=self.colors["HL"]
        )

        subtitle = Text(
            "A không phải ma trận vuông",
            font_size=30
        ).next_to(title, DOWN)

        subtitle2 = Text(
            "Không thể chéo hóa trực tiếp",
            font_size=30
        ).next_to(subtitle, DOWN)

        subtitle3 = Text(
            "Thay vào đó: Chéo hóa AᵀA",
            font_size=32,
            color=YELLOW
        ).next_to(subtitle2, DOWN)

        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.play(FadeIn(subtitle2))
        self.play(FadeIn(subtitle3))

        self.wait(3)

        self.clear_scene()

    def section2_build_ATA(self):

        title = Text(
            "Tạo ma trận vuông đối xứng AᵀA",
            font_size=36,
            color=self.colors["SIGMA"]
        ).to_edge(UP)

        A = Matrix(self.a_data).shift(LEFT*3)

        labelA = MathTex("A").next_to(A, UP)

        At = _transpose(self.a_data)
        ATA = _mat_mul(At, self.a_data)

        ATA_matrix = Matrix(
            np.round(ATA,2)
        ).shift(RIGHT*3)

        labelATA = MathTex("A^TA").next_to(ATA_matrix, UP)

        arrow = Arrow(
            A.get_right(),
            ATA_matrix.get_left()
        )

        explanation = Text(
            "Từ ma trận gầy → tạo ma trận vuông đối xứng",
            font_size=24
        ).to_edge(DOWN)

        self.play(Write(title))

        self.play(
            FadeIn(A),
            FadeIn(labelA)
        )

        self.play(
            GrowArrow(arrow),
            FadeIn(ATA_matrix),
            FadeIn(labelATA)
        )

        self.play(
            FadeIn(explanation)
        )

        self.wait(3)

        self.clear_scene()

    def section3_diagonalization(self):
        self.clear_scene()
        self.set_camera_orientation(phi=0, theta=-90*DEGREES)

        # --- 1. TIÊU ĐỀ CỐ ĐỊNH ---
        title = Text("Bước 3: Chéo hóa ma trận AᵀA", font_size=32, color=self.colors["HL"]).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        # --- TÍNH TOÁN DỮ LIỆU ---
        ATA_np = np.dot(self.a_data.T, self.a_data)
        eig_vals, eig_vecs = np.linalg.eigh(ATA_np)
        idx = eig_vals.argsort()[::-1]
        vals, vecs = eig_vals[idx], eig_vecs[:, idx]

        # --- 2. HIỂN THỊ AᵀA (TRUNG TÂM) ---
        ata_mat = Matrix(np.round(ATA_np, 0)).scale(0.8)
        ata_lbl = MathTex("A^TA =").next_to(ata_mat, LEFT)
        ata_group = VGroup(ata_lbl, ata_mat).center()

        self.play(FadeIn(ata_group, shift=UP))
        self.wait(1)

        # --- 3. BIẾN ĐỔI SANG ĐA THỨC ĐẶC TRƯNG ---
        step_desc = Text("1. Tìm Đa thức đặc trưng", font_size=24, color=GRAY_B)
        step_desc.next_to(title, DOWN, buff=0.3)
        char_eq = MathTex("\\det(A^TA - \\lambda I) = 0").next_to(ata_group, DOWN, buff=0.5)
        
        self.play(Write(step_desc), Write(char_eq))
        self.wait(2)

        # --- 4. BIẾN ĐỔI SANG TRỊ RIÊNG (EIGENVALUES) ---
        new_step_desc = Text("2. Giải phương trình tìm Trị riêng (λ)", font_size=24, color=GRAY_B)
        new_step_desc.next_to(title, DOWN, buff=0.3)
        eigen_vals_tex = MathTex(f"\\lambda_1 = {int(vals[0])}, \\quad \\lambda_2 = {int(vals[1])}").move_to(char_eq)

        self.play(
            ReplacementTransform(step_desc, new_step_desc),
            ReplacementTransform(char_eq, eigen_vals_tex)
        )
        self.wait(2)

        self.play(
            FadeOut(eigen_vals_tex),
            FadeOut(ata_group) # Nếu bạn còn giữ group A^TA ở trên
        )
        
        # STEP 3 — Tìm Vector riêng (Eigenvectors)
        step3_desc = Text("3. Tìm Vector riêng cho từng λ", font_size=24, color=GRAY_B)
        step3_desc.next_to(title, DOWN, buff=0.3)
        self.play(ReplacementTransform(new_step_desc, step3_desc))

        # Công thức tổng quát
        general_eq = MathTex("(A^TA - \\lambda I)\\mathbf{v} = \\mathbf{0}").shift(UP * 0.5)
        self.play(Write(general_eq))
        self.wait(1)

        # --- Tìm v1 cho lambda1 = 25 ---
        v1_logic = MathTex(
            "\\lambda_1 = 25 \\Rightarrow ", 
            "\\begin{bmatrix} 17-25 & 8 \\\\ 8 & 17-25 \\end{bmatrix}",
            "\\begin{bmatrix} x_1 \\\\ x_2 \\end{bmatrix} = \\mathbf{0}"
        ).scale(0.7).next_to(general_eq, DOWN, buff=0.5)
        
        v1_result = MathTex(
            "\\Rightarrow \\mathbf{v}_1 = \\begin{bmatrix} 0.71 \\\\ 0.71 \\end{bmatrix}", 
            color=self.colors["VT"]
        ).scale(0.7).next_to(v1_logic, RIGHT)

        self.play(Write(v1_logic))
        self.play(FadeIn(v1_result, shift=LEFT))
        self.wait(2)

        # --- Tìm v2 cho lambda2 = 9 ---
        v2_logic = MathTex(
            "\\lambda_2 = 9 \\Rightarrow ", 
            "\\begin{bmatrix} 17-9 & 8 \\\\ 8 & 17-9 \\end{bmatrix}",
            "\\begin{bmatrix} x_1 \\\\ x_2 \\end{bmatrix} = \\mathbf{0}"
        ).scale(0.7).move_to(v1_logic)
        
        v2_result = MathTex(
            "\\Rightarrow \\mathbf{v}_2 = \\begin{bmatrix} -0.71 \\\\ 0.71 \\end{bmatrix}", 
            color=self.colors["VT"]
        ).scale(0.7).next_to(v2_logic, RIGHT)

        # Biến đổi từ v1 sang v2 để tiết kiệm diện tích, không bị trùng
        self.play(
            ReplacementTransform(v1_logic, v2_logic),
            ReplacementTransform(v1_result, v2_result)
        )
        self.wait(2)

        # 1. Định nghĩa ma trận V cuối cùng
        v_final_mat = Matrix([[0.71, -0.71], [0.71, 0.71]]).set_color(self.colors["VT"]).scale(0.8)
        v_final_lbl = MathTex("V = [\\mathbf{v}_1 \\; \\mathbf{v}_2] =").next_to(v_final_mat, LEFT)
        v_group = VGroup(v_final_lbl, v_final_mat).center()

        # 2. Thực hiện xóa các phần nháp và biến hình
        # Chúng ta biến v2_result thành v_group để tạo cảm giác các vector hội tụ lại
        self.play(
            FadeOut(general_eq),   # Xóa công thức (A-lambda I)v=0
            FadeOut(v2_logic),     # Xóa phần tính toán lambda2
            FadeOut(step3_desc),   # Xóa dòng text "3. Tìm vector riêng"
            ReplacementTransform(v2_result, v_group) # Biến kết quả v2 thành ma trận V tổng quát
        )
        self.wait(2)

        # 3. Chuyển sang phần kết luận chéo hóa
        # Đẩy V lên hoặc thu nhỏ lại để lấy chỗ cho công thức A = VDV^T
        self.play(
            FadeOut(v_group)
        )

        # --- 5. HIỂN THỊ KẾT QUẢ CHÉO HÓA (PHÉP PHÂN TÍCH) ---
        # Di chuyển A^TA lên trên một chút để lấy chỗ cho PDP^-1
        self.play(ata_group.animate.shift(UP * 1.5).scale(0.7), FadeOut(eigen_vals_tex), FadeOut(new_step_desc))

        v_mat = Matrix(np.round(vecs, 2)).set_color(self.colors["VT"]).scale(0.6)
        d_mat = Matrix([[int(vals[0]), 0], [0, int(vals[1])]]).set_color(self.colors["SIGMA"]).scale(0.6)
        vt_mat = Matrix(np.round(vecs.T, 2)).set_color(self.colors["VT"]).scale(0.6)
        
        pdp_group = VGroup(v_mat, d_mat, vt_mat).arrange(RIGHT, buff=0.5).shift(DOWN * 0.5)
        pdp_lbl = MathTex("V", "D", "V^T").arrange(RIGHT, buff=1.6).next_to(pdp_group, UP, buff=0.2)
        pdp_lbl[0].set_color(self.colors["VT"])
        pdp_lbl[1].set_color(self.colors["SIGMA"])
        pdp_lbl[2].set_color(self.colors["VT"])

        equal_sign = MathTex("=").next_to(pdp_group, LEFT)

        self.play(
            FadeIn(pdp_group, shift=UP),
            FadeIn(pdp_lbl),
            FadeIn(equal_sign)
        )
        
        final_desc = Text("Phép phân tích Chéo hóa hoàn tất", font_size=22, color=GREEN).to_edge(DOWN, buff=0.7)
        self.play(Write(final_desc))
        self.wait(3)

    def section4_bridge_svd(self):
        self.clear_scene()
        # Đảm bảo tiêu đề vẫn ở đó
        title = Text("Cầu nối sang SVD", font_size=32, color=self.colors["HL"]).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)

        # Tọa độ dữ liệu
        vals = [25, 9] # Dựa trên ma trận của bạn
        sigmas = [5, 3]

        # Hiển thị Lambda và Sigma song song
        lambda_box = VGroup(
            Text("Eigenvalues (AᵀA)", font_size=20),
            MathTex(f"\\lambda_1 = {vals[0]}"),
            MathTex(f"\\lambda_2 = {vals[1]}")
        ).arrange(DOWN).shift(LEFT * 3)

        sigma_box = VGroup(
            Text("Singular Values (A)", font_size=20),
            MathTex(f"\\sigma_1 = \\sqrt{25} = {sigmas[0]}"),
            MathTex(f"\\sigma_2 = \\sqrt{9} = {sigmas[1]}")
        ).arrange(DOWN).shift(RIGHT * 3)

        arrow = DoubleArrow(lambda_box.get_right(), sigma_box.get_left(), color=YELLOW)
        formula = MathTex("\\sigma_i = \\sqrt{\\lambda_i}").next_to(arrow, UP)

        self.play(FadeIn(lambda_box))
        self.play(Create(arrow), Write(formula))
        self.play(FadeIn(sigma_box))

        conclusion = Text(
            "Kết luận: Chéo hóa AᵀA cung cấp thông tin cốt lõi cho SVD", 
            font_size=24, color=self.colors["HL"]
        ).to_edge(DOWN, buff=1)

        self.play(Write(conclusion))
        self.wait(4)


    # ----------------------------------------------------------
    # 10. ỨNG DỤNG SVD TRONG NÉN ẢNH
    # Low-Rank Approximation
    # ----------------------------------------------------------
    def section5_svd_application(self):
        self.clear_scene()
        
        # 1. TIÊU ĐỀ
        title = Text("Ứng dụng SVD - Nén dữ liệu & Khử nhiễu", 
                     font_size=30, color=YELLOW).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # 2. CÔNG THỨC (Đưa lên góc nhanh hơn để tập trung vào ảnh)
        formula = MathTex(
            "A_k", "=", "\\sum_{i=1}^k", "\\sigma_i", "\\mathbf{u}_i", "\\mathbf{v}_i^T"
        ).scale(0.7).to_corner(UL, buff=0.8).shift(DOWN*0.5)
        self.play(Write(formula))

        # =====================================================
        # DỮ LIỆU ẢNH (FIX ĐỂ THẤY RÕ SỰ THAY ĐỔI)
        # =====================================================
        H, W = 120, 120
        # Tạo một khối trắng sắc nét ở giữa trên nền đen
        A_clean = np.zeros((H, W))
        A_clean[30:90, 30:90] = 1.0
        
        # THÊM NHIỄU (Noise): Đây là chìa khóa để thấy k nhỏ bị mờ/nhiễu
        # Nếu k nhỏ, SVD sẽ bỏ qua nhiễu (vốn là các trị riêng nhỏ)
        np.random.seed(42)
        noise = np.random.normal(0, 0.4, (H, W))
        A_data = A_clean + noise

        U_val, S_val, Vt_val = np.linalg.svd(A_data, full_matrices=False)

        def get_svd_image(k_rank):
            # Tái thiết Ak
            ak_raw = np.dot(U_val[:, :k_rank], np.dot(np.diag(S_val[:k_rank]), Vt_val[:k_rank, :]))
            # Chuẩn hóa dải màu cực kỳ quan trọng
            ak_norm = np.interp(ak_raw, (ak_raw.min(), ak_raw.max()), (0, 255)).astype(np.uint8)
            # Dùng NEAREST (0) để thấy rõ khối pixel khi k thấp
            return ImageMobject(ak_norm).set_resampling_strategy(0).scale_to_fit_height(3.5)

        # 3. ẢNH GỐC - Dịch xuống (DOWN * 1.2) để né công thức hoàn toàn
        img_original = get_svd_image(H).shift(LEFT * 3.5 + DOWN * 1.2)
        lbl_original = Text("Dữ liệu gốc (Nhiễu)", font_size=18, color=BLUE).next_to(img_original, DOWN, buff=0.3)
        
        # 4. ẢNH NÉN - Dịch xuống tương ứng
        img_compressed = get_svd_image(1).shift(RIGHT * 3.5 + DOWN * 1.2)
        lbl_compressed = Text("Xấp xỉ hạng k", font_size=18, color=GREEN).next_to(img_compressed, DOWN, buff=0.3)
        k_display = MathTex("k = 1", color=YELLOW, font_size=28).next_to(lbl_compressed, RIGHT, buff=0.4)

        # Hiện ảnh và chữ
        self.play(FadeIn(img_original), Write(lbl_original))
        self.play(FadeIn(img_compressed), Write(lbl_compressed), Write(k_display))
        self.wait(1)

        # 5. LOOP TĂNG K: Quan sát ảnh mờ -> hiện khối -> hiện nhiễu
        # k=1: Mờ căm | k=5: Hiện khối vuông | k=20: Khá nét | k=H: Giống gốc
        for k in [2, 5, 15, 30, 120]:
            new_img = get_svd_image(k).move_to(img_compressed)
            new_k = MathTex(f"k = {k}", color=YELLOW, font_size=32).move_to(k_display)
            
            # Tăng run_time một chút ở các bước đầu để người xem kịp nhìn
            self.play(
                ReplacementTransform(img_compressed, new_img),
                ReplacementTransform(k_val_display if 'k_val_display' in locals() else k_display, new_k),
                run_time=1.2
            )
            img_compressed = new_img
            k_val_display = new_k
            self.wait(0.5)

        self.wait(2)

        # 6. DỌN DẸP & KẾT LUẬN
        all_mobs = [m for m in self.mobjects if m != title]
        self.play(*(FadeOut(m) for m in all_mobs))
        
        concl = Text("SVD tách biệt cấu trúc chính khỏi nhiễu vô nghĩa.", 
                     font_size=24, color=GREEN, t2c={"cấu trúc chính": YELLOW}).center()
        self.play(Write(concl))
        self.wait(3)


    def the_outro_scene(self):
        self.clear_scene()
        # 1. THÔNG ĐIỆP CUỐI
        summary_text = Text(
            "SVD không chỉ là một công thức...", 
            font_size=32, weight=LIGHT
        ).shift(UP * 0.5)
        
        highlight_text = Text(
            "Đó là cách toán học nhìn thấu sự hỗn loạn.",
            font_size=36, color=YELLOW, weight=BOLD
        ).next_to(summary_text, DOWN, buff=0.5)

        self.play(Write(summary_text))
        self.play(FadeIn(highlight_text, shift=UP))
        self.wait(2)

        # 2. HIỆU ỨNG BIẾN MẤT CINE
        # Tạo một vòng tròn tỏa sáng như thể đang "đóng" ống kính lại
        closing_circle = Circle(radius=0.1, color=WHITE, fill_opacity=1).set_stroke(width=0)
        self.play(
            ReplacementTransform(VGroup(summary_text, highlight_text), closing_circle),
            run_time=1
        )
        
        # Flash cuối cùng rồi biến mất vào hư không
        self.play(
            closing_circle.animate.scale(100).set_fill(opacity=0),
            run_time=0.8,
            rate_func=slow_into
        )
        
        # 3. LOGO HOẶC THANKS (Tùy chọn)
        thanks = Text("Cảm ơn bạn đã theo dõi!", font_size=24, color=GRAY_B)
        self.play(FadeIn(thanks))
        self.wait(2)
        self.play(FadeOut(thanks))
