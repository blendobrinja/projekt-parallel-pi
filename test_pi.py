#!/usr/bin/env python3
"""
Unit Tests für Parallel PI Berechnung
Testet die wichtigsten Funktionen und Berechnungen
"""

import unittest
import math
import sys
import os

# Pfad zum Hauptmodul hinzufügen
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pi import (
    PiBasisRechner,
    GILThreadRechner,
    ParalleleThreadRechner,
    ProzessRechner,
    ProducerConsumerRechner,
    MapReduceRechner,
    ThreadPoolRechner
)


class TestPiBasisRechner(unittest.TestCase):
    """Tests für die Basis-Funktionalität"""

    def setUp(self):
        self.rechner = PiBasisRechner()

    def test_leibniz_summand_erste_terme(self):
        """Teste die ersten Terme der Leibniz-Reihe"""
        # π/4 = 1 - 1/3 + 1/5 - 1/7 + ...
        self.assertAlmostEqual(self.rechner.leibniz_summand(0), 1.0)
        self.assertAlmostEqual(self.rechner.leibniz_summand(1), -1.0/3.0)
        self.assertAlmostEqual(self.rechner.leibniz_summand(2), 1.0/5.0)
        self.assertAlmostEqual(self.rechner.leibniz_summand(3), -1.0/7.0)

    def test_leibniz_bereich_kleine_werte(self):
        """Teste Bereichsberechnung mit bekannten Werten"""
        # Erstes Term: 1.0
        ergebnis = self.rechner.leibniz_bereich(0, 1)
        self.assertAlmostEqual(ergebnis, 1.0)

        # Erste zwei Terme: 1 - 1/3 = 2/3
        ergebnis = self.rechner.leibniz_bereich(0, 2)
        self.assertAlmostEqual(ergebnis, 2.0/3.0, places=5)

    def test_sequenzielle_berechnung_konvergenz(self):
        """Teste dass die Berechnung gegen π konvergiert"""
        pi_schaetzung, fehler, messung = self.rechner.sequenzielle_berechnung(10000)

        # Mit 10000 Termen sollten wir π auf ~3 Dezimalstellen genau haben
        self.assertLess(fehler, 0.001)  # Fehler < 0.1%
        self.assertGreater(pi_schaetzung, 3.0)
        self.assertLess(pi_schaetzung, 3.2)

        # Messdaten sollten plausibel sein
        self.assertEqual(messung['iterationen'], 10000)
        self.assertEqual(messung['methode'], 'sequenziell')
        self.assertGreater(messung['laufzeit'], 0)


class TestGILThreadRechner(unittest.TestCase):
    """Tests für GIL Thread Implementation (Hauptfeature)"""

    def setUp(self):
        self.rechner = GILThreadRechner()

    def test_gil_threads_grundfunktion(self):
        """Teste grundlegende GIL Thread Funktionalität"""
        pi_schaetzung, fehler, messung = self.rechner.berechne_mit_gil_threads(1000, 2)

        # Sollte ähnlich der sequenziellen Berechnung sein
        self.assertLess(fehler, 0.01)
        self.assertEqual(messung['methode'], 'gil_threads')
        self.assertEqual(messung['arbeiter'], 2)

    def test_gil_threads_verschiedene_thread_anzahlen(self):
        """Teste mit verschiedenen Thread-Anzahlen"""
        for threads in [1, 2, 4, 8]:
            with self.subTest(threads=threads):
                pi_schaetzung, fehler, messung = self.rechner.berechne_mit_gil_threads(5000, threads)

                self.assertLess(fehler, 0.005)
                self.assertEqual(messung['arbeiter'], threads)
                self.assertGreater(pi_schaetzung, 3.0)


class TestParallelisierungsVergleich(unittest.TestCase):
    """Vergleicht verschiedene Parallelisierungs-Ansätze"""

    def setUp(self):
        self.iterationen = 50000  # Groß genug für Genauigkeit, klein genug für Tests

    def test_alle_methoden_konvergieren(self):
        """Teste dass alle Methoden zu ähnlichen Ergebnissen konvergieren"""
        ergebnisse = {}

        # GIL Threads
        gil_rechner = GILThreadRechner()
        pi, fehler, _ = gil_rechner.berechne_mit_gil_threads(self.iterationen, 2)
        ergebnisse['gil'] = pi

        # Parallele Threads
        thread_rechner = ParalleleThreadRechner()
        pi, fehler, _ = thread_rechner.berechne_mit_parallelen_threads(self.iterationen, 2)
        ergebnisse['parallel'] = pi

        # Prozesse
        prozess_rechner = ProzessRechner()
        pi, fehler, _ = prozess_rechner.berechne_mit_prozessen(self.iterationen, 2)
        ergebnisse['prozess'] = pi

        # Producer/Consumer
        pc_rechner = ProducerConsumerRechner()
        pi, fehler, _ = pc_rechner.berechne_producer_consumer(self.iterationen, 2)
        ergebnisse['producer_consumer'] = pi

        # Map/Reduce
        mr_rechner = MapReduceRechner()
        pi, fehler, _ = mr_rechner.berechne_map_reduce(self.iterationen, 2)
        ergebnisse['map_reduce'] = pi

        # Thread Pool
        pool_rechner = ThreadPoolRechner()
        pi, fehler, _ = pool_rechner.berechne_mit_thread_pool(self.iterationen, 10)
        ergebnisse['thread_pool'] = pi

        # Alle Ergebnisse sollten sehr ähnlich sein (innerhalb 0.1% voneinander)
        werte = list(ergebnisse.values())
        max_wert = max(werte)
        min_wert = min(werte)
        relative_differenz = (max_wert - min_wert) / min_wert

        self.assertLess(relative_differenz, 0.001,
                       f"Zu große Unterschiede zwischen Methoden: {ergebnisse}")

    def test_performance_messung_plausibilitaet(self):
        """Teste dass Performance-Messungen plausibel sind"""
        gil_rechner = GILThreadRechner()

        # Kleine Berechnung
        _, _, messung_klein = gil_rechner.berechne_mit_gil_threads(1000, 2)

        # Große Berechnung
        _, _, messung_gross = gil_rechner.berechne_mit_gil_threads(100000, 2)

        # Größere Berechnung sollte länger dauern
        self.assertGreater(messung_gross['laufzeit'], messung_klein['laufzeit'])
        self.assertGreater(messung_gross['iterationen'], messung_klein['iterationen'])


class TestFehlerbehandlung(unittest.TestCase):
    """Tests für Edge Cases und Fehlerbehandlung"""

    def test_null_iterationen(self):
        """Teste Verhalten bei 0 Iterationen"""
        rechner = PiBasisRechner()
        pi_schaetzung, fehler, messung = rechner.sequenzielle_berechnung(0)

        self.assertEqual(pi_schaetzung, 0.0)
        self.assertEqual(messung['iterationen'], 0)

    def test_sehr_kleine_iterationen(self):
        """Teste Verhalten bei sehr wenigen Iterationen"""
        rechner = GILThreadRechner()
        pi_schaetzung, fehler, messung = rechner.berechne_mit_gil_threads(1, 1)

        # Mit nur 1 Iteration: π ≈ 4 * 1 = 4
        self.assertAlmostEqual(pi_schaetzung, 4.0)
        self.assertEqual(messung['iterationen'], 1)

    def test_mehr_threads_als_iterationen(self):
        """Teste was passiert wenn mehr Threads als Iterationen"""
        rechner = GILThreadRechner()
        pi_schaetzung, fehler, messung = rechner.berechne_mit_gil_threads(5, 10)

        # Sollte trotzdem funktionieren
        self.assertIsNotNone(pi_schaetzung)
        self.assertEqual(messung['arbeiter'], 10)


class TestMathematischeKorrektheit(unittest.TestCase):
    """Tests für mathematische Korrektheit der Leibniz-Formel"""

    def test_leibniz_formel_konvergenz(self):
        """Teste dass die Formel wirklich gegen π/4 konvergiert"""
        rechner = PiBasisRechner()

        # Test mit verschiedenen Iterationszahlen
        iterationen_liste = [1000, 10000, 100000]
        vorheriger_fehler = float('inf')

        for iterationen in iterationen_liste:
            pi_schaetzung, fehler, _ = rechner.sequenzielle_berechnung(iterationen)

            # π-Schätzung sollte im plausiblen Bereich sein
            self.assertGreater(pi_schaetzung, 2.5)
            self.assertLess(pi_schaetzung, 4.0)

            # Fehler sollte mit mehr Iterationen kleiner werden
            self.assertLess(fehler, vorheriger_fehler)
            vorheriger_fehler = fehler

    def test_leibniz_symmetrie(self):
        """Teste mathematische Eigenschaften der Leibniz-Reihe"""
        rechner = PiBasisRechner()

        # Gerade Indizes sind positiv
        self.assertGreater(rechner.leibniz_summand(0), 0)
        self.assertGreater(rechner.leibniz_summand(2), 0)
        self.assertGreater(rechner.leibniz_summand(4), 0)

        # Ungerade Indizes sind negativ
        self.assertLess(rechner.leibniz_summand(1), 0)
        self.assertLess(rechner.leibniz_summand(3), 0)
        self.assertLess(rechner.leibniz_summand(5), 0)


# Test Runner
if __name__ == '__main__':
    print("Starte Unit Tests für Parallel π Berechnung...")
    print("=" * 60)

    # Test Suite erstellen
    suite = unittest.TestSuite()

    # Basis Tests (wichtigste)
    suite.addTest(unittest.makeSuite(TestPiBasisRechner))
    suite.addTest(unittest.makeSuite(TestGILThreadRechner))

    # Vergleichs Tests
    suite.addTest(unittest.makeSuite(TestParallelisierungsVergleich))

    # Edge Case Tests
    suite.addTest(unittest.makeSuite(TestFehlerbehandlung))

    # Mathematische Tests
    suite.addTest(unittest.makeSuite(TestMathematischeKorrektheit))

    # Tests ausführen
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Zusammenfassung
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ Alle Tests erfolgreich!")
        print(f"Ausgeführt: {result.testsRun} Tests")
    else:
        print("❌ Einige Tests fehlgeschlagen!")
        print(f"Fehlgeschlagen: {len(result.failures)} von {result.testsRun}")
        print(f"Fehler: {len(result.errors)}")

    print("=" * 60)
