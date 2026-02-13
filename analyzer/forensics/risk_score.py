"""
PDF/Image Forensics Risk Scoring

Analyzes documents for potential edit artifacts and assigns a risk score (0-100%).
This is a heuristic assessment, not definitive proof of tampering.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ForensicsAnalyzer:
    """Analyzes PDFs for potential manipulation artifacts."""

    def __init__(self, dpi: int = 300):
        """
        Initialize forensics analyzer.

        Args:
            dpi: DPI for PDF rendering
        """
        self.dpi = dpi

        # Try to import required libraries
        try:
            import cv2
            from pdf2image import convert_from_path
            from PIL import Image
            self.cv2 = cv2
            self.convert_from_path = convert_from_path
            self.Image = Image
            self.available = True
        except ImportError as e:
            logger.warning(f"Forensics libraries not available: {e}")
            self.available = False

    def analyze(self, pdf_path: str, profile: Any) -> Dict[str, Any]:
        """
        Analyze PDF for manipulation artifacts.

        Args:
            pdf_path: Path to PDF file
            profile: Document profile with forensics config

        Returns:
            Dict with risk_score_percent and contributing signals
        """
        if not self.available:
            return {
                'risk_score_percent': 0,
                'signals': [],
                'error': 'forensics_libraries_unavailable'
            }

        try:
            forensics_config = profile.forensics if profile else {}
            if not forensics_config.get('enabled', True):
                return {
                    'risk_score_percent': 0,
                    'signals': [],
                    'message': 'forensics_disabled_in_profile'
                }

            # Convert PDF to images
            images = self.convert_from_path(
                pdf_path,
                dpi=forensics_config.get('dpi', self.dpi),
                fmt='RGB'
            )

            # Analyze each page
            all_signals = []
            page_scores = []

            for page_num, image in enumerate(images, 1):
                page_signals = self._analyze_page(
                    image,
                    page_num,
                    forensics_config
                )
                all_signals.extend(page_signals)

                # Calculate page risk score
                page_score = self._calculate_page_score(page_signals)
                page_scores.append(page_score)

            # Calculate overall risk score
            risk_score = self._calculate_overall_score(page_scores, all_signals)

            return {
                'risk_score_percent': risk_score,
                'signals': all_signals,
                'pages_analyzed': len(images),
                'max_page_score': max(page_scores) if page_scores else 0
            }

        except Exception as e:
            logger.error(f"Forensics analysis failed: {e}", exc_info=True)
            return {
                'risk_score_percent': 0,
                'signals': [],
                'error': str(e)
            }

    def _analyze_page(self,
                      image: Any,
                      page_num: int,
                      config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze a single page for artifacts.

        Args:
            image: PIL Image
            page_num: Page number
            config: Forensics configuration

        Returns:
            List of detected signals
        """
        signals = []

        # Convert PIL to numpy array
        img_array = np.array(image)

        # Convert to grayscale for analysis
        gray = self.cv2.cvtColor(img_array, self.cv2.COLOR_RGB2GRAY)

        # Signal 1: Local compression artifact detection
        if config.get('compression_analysis', True):
            compression_signals = self._detect_compression_artifacts(gray, page_num)
            signals.extend(compression_signals)

        # Signal 2: Noise inconsistency
        if config.get('noise_analysis', True):
            noise_signals = self._detect_noise_inconsistency(gray, page_num)
            signals.extend(noise_signals)

        # Signal 3: Edge anomalies (near numeric regions if specified)
        if config.get('analyze_numeric_regions', True):
            edge_signals = self._detect_edge_anomalies(gray, page_num)
            signals.extend(edge_signals)

        return signals

    def _detect_compression_artifacts(self,
                                     gray: np.ndarray,
                                     page_num: int) -> List[Dict[str, Any]]:
        """
        Detect local compression inconsistencies.

        Args:
            gray: Grayscale image
            page_num: Page number

        Returns:
            List of compression artifact signals
        """
        signals = []

        try:
            # Divide image into blocks
            h, w = gray.shape
            block_size = 64  # 64x64 pixel blocks
            compression_scores = []

            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray[y:y+block_size, x:x+block_size]

                    # Measure compression artifacts using DCT
                    # High-frequency components indicate potential compression
                    dct = self.cv2.dct(np.float32(block))
                    high_freq = np.sum(np.abs(dct[32:, 32:]))  # Bottom-right quadrant
                    compression_scores.append((high_freq, x, y))

            # Find outliers (blocks with unusual compression)
            scores = [s[0] for s in compression_scores]
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            # Flag blocks that are more than 2 std devs from mean
            outliers = [(s, x, y) for s, x, y in compression_scores
                       if abs(s - mean_score) > 2 * std_score]

            if outliers:
                # Calculate signal strength
                max_deviation = max(abs(s - mean_score) for s, _, _ in outliers)
                weight = min(max_deviation / (3 * std_score), 1.0) * 25  # Max 25 points

                signals.append({
                    'type': 'compression_inconsistency',
                    'page': page_num,
                    'weight': weight,
                    'description': f"Found {len(outliers)} blocks with unusual compression",
                    'regions': len(outliers)
                })

        except Exception as e:
            logger.debug(f"Compression analysis failed on page {page_num}: {e}")

        return signals

    def _detect_noise_inconsistency(self,
                                   gray: np.ndarray,
                                   page_num: int) -> List[Dict[str, Any]]:
        """
        Detect inconsistent noise patterns (potential copy-paste).

        Args:
            gray: Grayscale image
            page_num: Page number

        Returns:
            List of noise inconsistency signals
        """
        signals = []

        try:
            # Measure local noise variance in different regions
            h, w = gray.shape
            region_size = 128
            noise_levels = []

            for y in range(0, h - region_size, region_size):
                for x in range(0, w - region_size, region_size):
                    region = gray[y:y+region_size, x:x+region_size]

                    # Calculate local standard deviation (noise level)
                    local_std = np.std(region)
                    noise_levels.append(local_std)

            # Check for high variance in noise levels
            noise_variance = np.var(noise_levels)
            noise_std = np.std(noise_levels)

            # If noise varies significantly, it might indicate manipulation
            if noise_std > 5:  # Threshold for unusual noise variation
                weight = min(noise_std / 10, 1.0) * 20  # Max 20 points

                signals.append({
                    'type': 'noise_inconsistency',
                    'page': page_num,
                    'weight': weight,
                    'description': f"Inconsistent noise patterns detected (std={noise_std:.2f})",
                    'noise_std': float(noise_std)
                })

        except Exception as e:
            logger.debug(f"Noise analysis failed on page {page_num}: {e}")

        return signals

    def _detect_edge_anomalies(self,
                              gray: np.ndarray,
                              page_num: int) -> List[Dict[str, Any]]:
        """
        Detect unusual edge patterns that might indicate text editing.

        Args:
            gray: Grayscale image
            page_num: Page number

        Returns:
            List of edge anomaly signals
        """
        signals = []

        try:
            # Apply edge detection
            edges = self.cv2.Canny(gray, 50, 150)

            # Find contours
            contours, _ = self.cv2.findContours(edges, self.cv2.RETR_EXTERNAL,
                                                self.cv2.CHAIN_APPROX_SIMPLE)

            # Look for small, isolated edge regions (potential text edits)
            suspicious_contours = []
            for contour in contours:
                area = self.cv2.contourArea(contour)
                if 50 < area < 500:  # Small regions
                    x, y, w, h = self.cv2.boundingRect(contour)
                    # Check if this region has sharp, artificial-looking edges
                    suspicious_contours.append((x, y, w, h, area))

            if len(suspicious_contours) > 20:  # Many small suspicious regions
                weight = min(len(suspicious_contours) / 50, 1.0) * 15  # Max 15 points

                signals.append({
                    'type': 'edge_anomalies',
                    'page': page_num,
                    'weight': weight,
                    'description': f"Found {len(suspicious_contours)} suspicious edge regions",
                    'regions': len(suspicious_contours)
                })

        except Exception as e:
            logger.debug(f"Edge analysis failed on page {page_num}: {e}")

        return signals

    def _calculate_page_score(self, signals: List[Dict[str, Any]]) -> float:
        """Calculate risk score for a single page."""
        if not signals:
            return 0.0

        # Sum weights, cap at 100
        total_weight = sum(s['weight'] for s in signals)
        return min(total_weight, 100.0)

    def _calculate_overall_score(self,
                                page_scores: List[float],
                                all_signals: List[Dict[str, Any]]) -> int:
        """
        Calculate overall document risk score.

        Args:
            page_scores: List of per-page scores
            all_signals: All detected signals

        Returns:
            Risk score as integer (0-100)
        """
        if not page_scores:
            return 0

        # Use max page score as baseline
        max_score = max(page_scores)

        # Adjust based on how many pages have issues
        pages_with_issues = sum(1 for score in page_scores if score > 10)
        issue_ratio = pages_with_issues / len(page_scores)

        # If multiple pages have issues, increase confidence
        adjusted_score = max_score * (0.7 + 0.3 * issue_ratio)

        return int(min(adjusted_score, 100))
