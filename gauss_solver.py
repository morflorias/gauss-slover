"""
Модуль для решения систем линейных алгебраических уравнений (СЛАУ) методом Гаусса.
Вывод переработан для создания строгого и лаконичного математического отчета.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


class SolutionStatus(Enum):
    """Статус решения системы линейных уравнений."""
    UNIQUE_SOLUTION = "Единственное решение"
    NO_SOLUTION = "Решений нет (система несовместна)"
    INFINITE_SOLUTIONS = "Бесконечное множество решений"


@dataclass
class SolutionResult:
    """Результат решения СЛАУ."""
    status: SolutionStatus
    solution: Optional[np.ndarray] = None
    rank: int = 0
    steps: List[str] = None
    error_message: str = ""

    def __post_init__(self):
        if self.steps is None:
            self.steps =[]


class GaussSolver:
    """Класс для решения систем линейных алгебраических уравнений методом Гаусса."""

    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
        self.steps: List[str] =[]

    def solve(self, A: np.ndarray, b: np.ndarray) -> SolutionResult:
        self.steps =[]

        if A.ndim != 2 or b.ndim != 1:
            return SolutionResult(status=SolutionStatus.NO_SOLUTION, error_message="Неверная размерность")
        n = A.shape[0]
        if A.shape[1] != n or b.shape[0] != n:
            return SolutionResult(status=SolutionStatus.NO_SOLUTION, error_message="Неверная размерность")

        augmented = np.column_stack([A.copy().astype(float), b.copy().astype(float)])
        self.steps.append("Исходная расширенная матрица:")
        for row in self._format_augmented_matrix_rows(augmented):
            self.steps.append(row)
        self.steps.append("")

        rank = self._forward_elimination(augmented, n)
        status, solution = self._check_and_solve(augmented, n, rank)

        return SolutionResult(status=status, solution=solution, rank=rank, steps=self.steps.copy())

    def _forward_elimination(self, augmented: np.ndarray, n: int) -> int:
        self.steps.append("ПРЯМОЙ ХОД")
        rank = 0

        for col in range(n):
            step_operations =[]

            # Поиск ведущего элемента
            max_row = col
            max_val = abs(augmented[col, col])
            for row in range(col + 1, n):
                if abs(augmented[row, col]) > max_val:
                    max_val = abs(augmented[row, col])
                    max_row = row

            if max_row != col:
                augmented[[col, max_row]] = augmented[[max_row, col]]
                step_operations.append(f"R{col + 1} ↔ R{max_row + 1}")

            pivot = augmented[col, col]
            if abs(pivot) < self.epsilon:
                continue

            rank += 1

            if abs(pivot - 1.0) > self.epsilon:
                augmented[col] = augmented[col] / pivot
                step_operations.append(f"R{col + 1} = R{col + 1} / {pivot:.4f}")

            for row in range(col + 1, n):
                factor = augmented[row, col]
                if abs(factor) > self.epsilon:
                    augmented[row] = augmented[row] - factor * augmented[col]
                    sign = "+" if factor < 0 else "-"
                    step_operations.append(f"R{row + 1} = R{row + 1} {sign} {abs(factor):.4f} * R{col + 1}")

            if step_operations:
                self.steps.append(f"Шаг {col + 1}")
                for op in step_operations:
                    self.steps.append(op)
                for row in self._format_augmented_matrix_rows(augmented):
                    self.steps.append(row)
                self.steps.append("")

        return rank

    def _check_and_solve(self, augmented: np.ndarray, n: int, rank: int) -> Tuple[SolutionStatus, Optional[np.ndarray]]:
        self.steps.append("ОБРАТНЫЙ ХОД И АНАЛИЗ")

        augmented_rank = rank
        for row in range(rank, n):
            if abs(augmented[row, n]) > self.epsilon:
                augmented_rank = row + 1
                break

        if augmented_rank > rank:
            for row in range(rank, n):
                if abs(augmented[row, n]) > self.epsilon:
                    self.steps.append(f"Противоречие в строке {row + 1}: 0 = {augmented[row, n]:.4f}")
                    break
            self.steps.append("Система несовместна")
            return SolutionStatus.NO_SOLUTION, None

        if rank < n:
            self.steps.append(f"Ранг ({rank}) < числа неизвестных ({n})")
            self.steps.append("Бесконечное множество решений")
            return SolutionStatus.INFINITE_SOLUTIONS, None

        solution = np.zeros(n)
        for i in range(n - 1, -1, -1):
            sum_val = sum(augmented[i, j] * solution[j] for j in range(i + 1, n))
            solution[i] = augmented[i, n] - sum_val

            if i == n - 1:
                self.steps.append(f"x[{i + 1}] = {augmented[i, n]:.4f}")
            else:
                self.steps.append(f"x[{i + 1}] = {augmented[i, n]:.4f} - ({sum_val:.4f}) = {solution[i]:.4f}")

        return SolutionStatus.UNIQUE_SOLUTION, solution

    def _format_augmented_matrix_rows(self, matrix: np.ndarray) -> List[str]:
        lines = []
        n = matrix.shape[1] - 1
        for row in matrix:
            left = " ".join(f"{val:10.4f}" for val in row[:n])
            right = f"{row[n]:10.4f}"
            lines.append(f"( {left} | {right} )")
        return lines

    @staticmethod
    def verify_solution(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> Tuple[bool, np.ndarray, float]:
        residual = A @ x - b
        max_error = np.max(np.abs(residual))
        is_correct = max_error < 1e-6
        return is_correct, residual, max_error