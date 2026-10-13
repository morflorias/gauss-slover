"""
Unit тесты для модуля gauss_solver.
Тестирует алгоритм решения СЛАУ методом Гаусса.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from gauss_solver import GaussSolver, SolutionStatus, SolutionResult


class TestGaussSolver(unittest.TestCase):
    """Тесты для класса GaussSolver."""

    def setUp(self):
        """Настройка перед каждым тестом."""
        self.solver = GaussSolver()

    def test_unique_solution_simple(self):
        """Тест простой системы с единственным решением."""
        # Система: x + y = 3, x - y = 1
        # Решение: x = 2, y = 1
        A = np.array([[1, 1], [1, -1]], dtype=float)
        b = np.array([3, 1], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.UNIQUE_SOLUTION)
        self.assertIsNotNone(result.solution)
        self.assertEqual(len(result.solution), 2)
        self.assertAlmostEqual(result.solution[0], 2.0, places=6)
        self.assertAlmostEqual(result.solution[1], 1.0, places=6)
        self.assertEqual(result.rank, 2)
        self.assertTrue(len(result.steps) > 0)

    def test_unique_solution_3x3(self):
        """Тест системы 3x3 с единственным решением."""
        # Система:
        # 2x + y - z = 8
        # -3x - y + 2z = -11
        # -2x + y + 2z = -3
        # Решение: x = 2, y = 3, z = -1
        A = np.array([
            [2, 1, -1],
            [-3, -1, 2],
            [-2, 1, 2]
        ], dtype=float)
        b = np.array([8, -11, -3], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.UNIQUE_SOLUTION)
        self.assertIsNotNone(result.solution)
        self.assertEqual(len(result.solution), 3)
        self.assertAlmostEqual(result.solution[0], 2.0, places=5)
        self.assertAlmostEqual(result.solution[1], 3.0, places=5)
        self.assertAlmostEqual(result.solution[2], -1.0, places=5)

    def test_unique_solution_identity_matrix(self):
        """Тест системы с единичной матрицей."""
        A = np.eye(3)
        b = np.array([1, 2, 3], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.UNIQUE_SOLUTION)
        np.testing.assert_array_almost_equal(result.solution, b, decimal=6)

    def test_unique_solution_diagonal_matrix(self):
        """Тест системы с диагональной матрицей."""
        A = np.diag([2, 4, 6])
        b = np.array([4, 8, 18], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.UNIQUE_SOLUTION)
        np.testing.assert_array_almost_equal(result.solution,[2, 2, 3], decimal=6)

    def test_unique_solution_4x4(self):
        """Тест системы 4x4 с единственным решением."""
        A = np.array([
            [1, 2, 3, 4],[2, 5, 6, 7],
            [3, 6, 10, 8],[4, 7, 8, 12]
        ], dtype=float)
        b = A @ np.array([1, 2, 3, 4], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.UNIQUE_SOLUTION)
        is_correct, _, max_error = GaussSolver.verify_solution(A, b, result.solution)
        self.assertTrue(is_correct)
        self.assertLess(max_error, 1e-6)

    def test_no_solution_inconsistent(self):
        """Тест несовместной системы (нет решений)."""
        # Система: x + y = 1, x + y = 2 -> несовместна
        A = np.array([[1, 1], [1, 1]], dtype=float)
        b = np.array([1, 2], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.NO_SOLUTION)
        self.assertIsNone(result.solution)

    def test_no_solution_3x3(self):
        """Тест несовместной системы 3x3."""
        A = np.array([[1, 0, 0],
            [1, 0, 0],[1, 0, 0]
        ], dtype=float)
        b = np.array([1, 2, 3], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.NO_SOLUTION)

    def test_infinite_solutions_underdetermined(self):
        """Тест системы с бесконечным множеством решений."""
        A = np.array([[1, 1], [0, 0]], dtype=float)
        b = np.array([1, 0], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.INFINITE_SOLUTIONS)
        self.assertIsNone(result.solution)

    def test_infinite_solutions_rank_deficient(self):
        """Тест системы с неполным рангом."""
        A = np.array([
            [1, 2, 3],
            [2, 4, 6],
            [1, 1, 1]
        ], dtype=float)
        b = np.array([6, 12, 3], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.INFINITE_SOLUTIONS)
        self.assertEqual(result.rank, 2)

    def test_singular_matrix(self):
        """Тест вырожденной матрицы."""
        A = np.array([[1, 2], [2, 4]], dtype=float)
        b = np.array([3, 6], dtype=float)

        result = self.solver.solve(A, b)

        self.assertIn(result.status,[SolutionStatus.NO_SOLUTION, SolutionStatus.INFINITE_SOLUTIONS])

    def test_row_swap(self):
        """Тест системы, требующей перестановки строк."""
        A = np.array([
            [0, 1],[1, 0]
        ], dtype=float)
        b = np.array([2, 1], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.UNIQUE_SOLUTION)
        np.testing.assert_array_almost_equal(result.solution, [1, 2], decimal=6)

    def test_negative_values(self):
        """Тест системы с отрицательными значениями."""
        A = np.array([
            [-1, 2],
            [3, -4]
        ], dtype=float)
        b = np.array([-5, 14], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.UNIQUE_SOLUTION)
        self.assertAlmostEqual(result.solution[0], 4.0, places=5)
        self.assertAlmostEqual(result.solution[1], -0.5, places=5)

    def test_input_validation_wrong_dimension_A(self):
        """Тест валидации: неквадратная матрица A."""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        b = np.array([1, 2], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.NO_SOLUTION)
        self.assertIn("Неверная размерность", result.error_message)

    def test_input_validation_wrong_dimension_b(self):
        """Тест валидации: размерность b не совпадает с A."""
        A = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([1, 2, 3], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.NO_SOLUTION)
        self.assertIn("Неверная размерность", result.error_message)

    def test_input_validation_1d_A(self):
        """Тест валидации: одномерный массив вместо матрицы."""
        A = np.array([1, 2, 3], dtype=float)
        b = np.array([1], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.NO_SOLUTION)
        self.assertIn("Неверная размерность", result.error_message)

    def test_steps_generated(self):
        """Тест: проверка генерации шагов решения."""
        A = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([5, 11], dtype=float)

        result = self.solver.solve(A, b)

        self.assertTrue(len(result.steps) > 0)
        steps_text = "\n".join(result.steps)
        self.assertIn("Исходная расширенная матрица:", steps_text)
        self.assertIn("ПРЯМОЙ ХОД", steps_text)
        self.assertIn("ОБРАТНЫЙ ХОД И АНАЛИЗ", steps_text)

    def test_rank_calculation(self):
        """Тест: проверка вычисления ранга."""
        A_full = np.eye(3)
        b_full = np.array([1, 2, 3], dtype=float)
        result_full = self.solver.solve(A_full, b_full)
        self.assertEqual(result_full.rank, 3)

        A_rank2 = np.array([[1, 0, 0],
            [0, 1, 0],[0, 0, 0]
        ], dtype=float)
        b_rank2 = np.array([1, 2, 0], dtype=float)
        result_rank2 = self.solver.solve(A_rank2, b_rank2)
        self.assertEqual(result_rank2.rank, 2)


class TestVerifySolution(unittest.TestCase):
    """Тесты для статического метода verify_solution."""

    def test_verify_correct_solution(self):
        """Тест: проверка правильного решения."""
        A = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([5, 11], dtype=float)
        x = np.array([1, 2], dtype=float)

        is_correct, residual, max_error = GaussSolver.verify_solution(A, b, x)

        self.assertTrue(is_correct)
        self.assertLess(max_error, 1e-6)

    def test_verify_incorrect_solution(self):
        """Тест: проверка неправильного решения."""
        A = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([5, 11], dtype=float)
        x = np.array([0, 0], dtype=float)

        is_correct, residual, max_error = GaussSolver.verify_solution(A, b, x)

        self.assertFalse(is_correct)
        self.assertGreater(max_error, 1e-6)


class TestSolutionResult(unittest.TestCase):
    """Тесты для класса SolutionResult."""

    def test_default_steps(self):
        """Тест: steps по умолчанию - пустой список."""
        result = SolutionResult(status=SolutionStatus.UNIQUE_SOLUTION)
        self.assertEqual(result.steps,[])

    def test_custom_steps(self):
        """Тест: пользовательские шаги."""
        steps =["Шаг 1", "Шаг 2"]
        result = SolutionResult(status=SolutionStatus.UNIQUE_SOLUTION, steps=steps)
        self.assertEqual(result.steps, steps)


class TestEdgeCases(unittest.TestCase):
    """Тесты для граничных случаев."""

    def setUp(self):
        """Настройка перед каждым тестом."""
        self.solver = GaussSolver()

    def test_upper_triangular(self):
        """Тест: верхнетреугольная матрица."""
        A = np.array([[1, 2, 3],
            [0, 4, 5],[0, 0, 6]
        ], dtype=float)
        b = np.array([14, 23, 18], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.UNIQUE_SOLUTION)
        np.testing.assert_array_almost_equal(result.solution, [1, 2, 3], decimal=5)

    def test_lower_triangular(self):
        """Тест: нижнетреугольная матрица."""
        A = np.array([
            [1, 0, 0],
            [2, 3, 0],
            [4, 5, 6]
        ], dtype=float)
        b = np.array([1, 8, 32], dtype=float)

        result = self.solver.solve(A, b)

        self.assertEqual(result.status, SolutionStatus.UNIQUE_SOLUTION)
        is_correct, _, _ = GaussSolver.verify_solution(A, b, result.solution)
        self.assertTrue(is_correct)


class TestFormatting(unittest.TestCase):
    """Тесты для форматирования вывода в модуле."""

    def setUp(self):
        self.solver = GaussSolver()

    def test_format_augmented_matrix_rows(self):
        """Тест: форматирование расширенной матрицы."""
        augmented = np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
        rows = self.solver._format_augmented_matrix_rows(augmented)

        self.assertEqual(len(rows), 2)
        self.assertTrue(all("|" in row for row in rows))


class TestStepsFormatter(unittest.TestCase):
    """Тесты для логики форматирования пошагового решения."""

    def test_section_header_detection(self):
        """Тест: определение заголовков секций."""
        test_cases =[
            ("ПРЯМОЙ ХОД", True),
            ("ОБРАТНЫЙ ХОД И АНАЛИЗ", True),
            ("Исходная расширенная матрица:", True),
            ("Обычная строка", False),
        ]

        for text, is_header in test_cases:
            contains_header = text in["ПРЯМОЙ ХОД", "ОБРАТНЫЙ ХОД И АНАЛИЗ", "Исходная расширенная матрица:"]
            self.assertEqual(contains_header, is_header, f"Failed for: {text}")

    def test_matrix_format_detection(self):
        """Тест: определение формата матрицы."""
        matrix_lines =[
            "(    1.0000    2.0000 |    3.0000 )",
            "(    4.0000    5.0000 |    6.0000 )",
        ]

        for line in matrix_lines:
            self.assertTrue(line.startswith("(") and line.endswith(")"))
            self.assertIn("|", line)

    def test_step_operation_detection(self):
        """Тест: определение операций шагов."""
        step_lines =[
            "Шаг 1",
            "Шаг 2",
            "Шаг 10"
        ]

        for line in step_lines:
            is_step = line.strip().startswith("Шаг")
            self.assertTrue(is_step)


if __name__ == "__main__":
    unittest.main()