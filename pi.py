#!/usr/bin/env python3
"""
Parallel PI Berechnung mit Leibniz Formel
Implementierung verschiedener Parallelisierungsansätze

π/4 = 1 - 1/3 + 1/5 - 1/7 + 1/9 - ... = Σ((-1)^k)/(2k+1)

Implementiert für Hochschulprojekt
"""

import argparse
import time
import math
import threading
import multiprocessing
from multiprocessing import Pool, Process
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
import socket
import pickle
import sys


class PiBasisRechner:
    """Grundklasse für PI-Berechnungen"""

    def __init__(self):
        self.ergebnisse = []
        self.zeitmessungen = {}

    def leibniz_summand(self, k):
        """Berechnet einen einzelnen Summanden der Leibniz-Reihe"""
        return (-1) ** k / (2 * k + 1)

    def leibniz_bereich(self, start, ende):
        """Berechnet Leibniz-Summe für einen bestimmten Bereich"""
        summe = 0.0
        for k in range(start, ende):
            summe += self.leibniz_summand(k)
        return summe

    def sequenzielle_berechnung(self, iterationen):
        """Einfache sequenzielle Berechnung als Referenz"""
        startzeit = time.time()

        pi_viertel = 0.0
        for k in range(iterationen):
            pi_viertel += self.leibniz_summand(k)

        pi_schaetzung = 4 * pi_viertel
        endzeit = time.time()

        laufzeit = endzeit - startzeit
        fehler = abs(pi_schaetzung - math.pi)

        messdaten = {
            'methode': 'sequenziell',
            'iterationen': iterationen,
            'laufzeit': laufzeit,
            'pi_wert': pi_schaetzung,
            'fehler': fehler,
            'arbeiter': 1
        }

        return pi_schaetzung, fehler, messdaten


class GILThreadRechner(PiBasisRechner):
    """PI-Berechnung mit Python GIL Threads"""

    def berechne_mit_gil_threads(self, iterationen, anzahl_threads=4):
        """Hauptmethode für GIL Thread Berechnung"""
        startzeit = time.time()

        # Arbeit auf Threads aufteilen
        chunk_groesse = iterationen // anzahl_threads
        threads = []
        ergebnisse = [0.0] * anzahl_threads

        def arbeiter_funktion(thread_id, start, ende):
            ergebnisse[thread_id] = self.leibniz_bereich(start, ende)

        # Threads erstellen und starten
        for i in range(anzahl_threads):
            start = i * chunk_groesse
            ende = (i + 1) * chunk_groesse if i < anzahl_threads - 1 else iterationen
            t = threading.Thread(target=arbeiter_funktion, args=(i, start, ende))
            threads.append(t)
            t.start()

        # Auf alle Threads warten
        for t in threads:
            t.join()

        pi_viertel = sum(ergebnisse)
        pi_schaetzung = 4 * pi_viertel

        endzeit = time.time()
        laufzeit = endzeit - startzeit
        fehler = abs(pi_schaetzung - math.pi)

        messdaten = {
            'methode': 'gil_threads',
            'iterationen': iterationen,
            'laufzeit': laufzeit,
            'pi_wert': pi_schaetzung,
            'fehler': fehler,
            'arbeiter': anzahl_threads
        }

        return pi_schaetzung, fehler, messdaten


class ParalleleThreadRechner(PiBasisRechner):
    """PI-Berechnung mit parallelen Threads (ThreadPoolExecutor)"""

    def berechne_mit_parallelen_threads(self, iterationen, anzahl_threads=4):
        """Berechnung mit ThreadPoolExecutor für bessere Parallelisierung"""
        startzeit = time.time()

        chunk_groesse = iterationen // anzahl_threads

        with ThreadPoolExecutor(max_workers=anzahl_threads) as executor:
            aufgaben = []
            for i in range(anzahl_threads):
                start = i * chunk_groesse
                ende = (i + 1) * chunk_groesse if i < anzahl_threads - 1 else iterationen
                aufgabe = executor.submit(self.leibniz_bereich, start, ende)
                aufgaben.append(aufgabe)

            ergebnisse = [aufgabe.result() for aufgabe in aufgaben]

        pi_viertel = sum(ergebnisse)
        pi_schaetzung = 4 * pi_viertel

        endzeit = time.time()
        laufzeit = endzeit - startzeit
        fehler = abs(pi_schaetzung - math.pi)

        messdaten = {
            'methode': 'parallele_threads',
            'iterationen': iterationen,
            'laufzeit': laufzeit,
            'pi_wert': pi_schaetzung,
            'fehler': fehler,
            'arbeiter': anzahl_threads
        }

        return pi_schaetzung, fehler, messdaten


class ProzessRechner(PiBasisRechner):
    """PI-Berechnung mit separaten Prozessen"""

    def berechne_mit_prozessen(self, iterationen, anzahl_prozesse=4):
        """Berechnung mit Multiprocessing für echte Parallelisierung"""
        startzeit = time.time()

        chunk_groesse = iterationen // anzahl_prozesse

        with ProcessPoolExecutor(max_workers=anzahl_prozesse) as executor:
            aufgaben = []
            for i in range(anzahl_prozesse):
                start = i * chunk_groesse
                ende = (i + 1) * chunk_groesse if i < anzahl_prozesse - 1 else iterationen
                aufgabe = executor.submit(self.leibniz_bereich, start, ende)
                aufgaben.append(aufgabe)

            ergebnisse = [aufgabe.result() for aufgabe in aufgaben]

        pi_viertel = sum(ergebnisse)
        pi_schaetzung = 4 * pi_viertel

        endzeit = time.time()
        laufzeit = endzeit - startzeit
        fehler = abs(pi_schaetzung - math.pi)

        messdaten = {
            'methode': 'prozesse',
            'iterationen': iterationen,
            'laufzeit': laufzeit,
            'pi_wert': pi_schaetzung,
            'fehler': fehler,
            'arbeiter': anzahl_prozesse
        }

        return pi_schaetzung, fehler, messdaten


class ProducerConsumerRechner(PiBasisRechner):
    """Producer/Consumer Architektur für PI-Berechnung"""

    def berechne_producer_consumer(self, iterationen, anzahl_consumer=4):
        """Implementierung des Producer/Consumer Patterns"""
        startzeit = time.time()

        aufgaben_queue = Queue()
        ergebnis_queue = Queue()
        chunk_groesse = 1000

        def producer():
            """Erstellt Arbeitsaufgaben und legt sie in die Queue"""
            for start in range(0, iterationen, chunk_groesse):
                ende = min(start + chunk_groesse, iterationen)
                aufgaben_queue.put((start, ende))

            # Stopp-Signale für Consumer
            for _ in range(anzahl_consumer):
                aufgaben_queue.put(None)

        def consumer():
            """Nimmt Aufgaben aus der Queue und berechnet Teilergebnisse"""
            while True:
                aufgabe = aufgaben_queue.get()
                if aufgabe is None:
                    break
                start, ende = aufgabe
                teilsumme = self.leibniz_bereich(start, ende)
                ergebnis_queue.put(teilsumme)
                aufgaben_queue.task_done()

        # Producer starten
        producer_thread = threading.Thread(target=producer)
        producer_thread.start()

        # Consumer starten
        consumer_threads = []
        for _ in range(anzahl_consumer):
            t = threading.Thread(target=consumer)
            t.start()
            consumer_threads.append(t)

        # Warten bis Producer fertig ist
        producer_thread.join()

        # Warten bis alle Consumer fertig sind
        for t in consumer_threads:
            t.join()

        # Ergebnisse sammeln
        teilsummen = []
        while not ergebnis_queue.empty():
            teilsummen.append(ergebnis_queue.get())

        pi_viertel = sum(teilsummen)
        pi_schaetzung = 4 * pi_viertel

        endzeit = time.time()
        laufzeit = endzeit - startzeit
        fehler = abs(pi_schaetzung - math.pi)

        messdaten = {
            'methode': 'producer_consumer',
            'iterationen': iterationen,
            'laufzeit': laufzeit,
            'pi_wert': pi_schaetzung,
            'fehler': fehler,
            'arbeiter': anzahl_consumer + 1
        }

        return pi_schaetzung, fehler, messdaten


class MapReduceRechner(PiBasisRechner):
    """Map/Filter/Reduce Implementierung"""

    def berechne_map_reduce(self, iterationen, anzahl_arbeiter=4):
        """Funktionale Programmierung mit Map/Reduce Pattern"""
        startzeit = time.time()

        chunk_groesse = iterationen // anzahl_arbeiter

        # Map Phase: Arbeit verteilen
        bereiche = []
        for i in range(anzahl_arbeiter):
            start = i * chunk_groesse
            ende = (i + 1) * chunk_groesse if i < anzahl_arbeiter - 1 else iterationen
            bereiche.append((start, ende))

        # Map Phase ausführen
        with ThreadPoolExecutor(max_workers=anzahl_arbeiter) as executor:
            # Map: Teilberechnungen durchführen
            map_ergebnisse = list(executor.map(
                lambda bereich: self.leibniz_bereich(bereich[0], bereich[1]),
                bereiche
            ))

        # Filter Phase: Gültige Ergebnisse filtern
        gefilterte_ergebnisse = [erg for erg in map_ergebnisse if erg is not None]

        # Reduce Phase: Alle Teilergebnisse zusammenfassen
        pi_viertel = sum(gefilterte_ergebnisse)
        pi_schaetzung = 4 * pi_viertel

        endzeit = time.time()
        laufzeit = endzeit - startzeit
        fehler = abs(pi_schaetzung - math.pi)

        messdaten = {
            'methode': 'map_reduce',
            'iterationen': iterationen,
            'laufzeit': laufzeit,
            'pi_wert': pi_schaetzung,
            'fehler': fehler,
            'arbeiter': anzahl_arbeiter
        }

        return pi_schaetzung, fehler, messdaten


class ThreadPoolRechner(PiBasisRechner):
    """Thread Pool Implementierung"""

    def berechne_mit_thread_pool(self, iterationen, pool_groesse=100):
        """Berechnung mit vordefiniertem Thread Pool"""
        startzeit = time.time()

        chunk_groesse = max(1, iterationen // pool_groesse)

        with ThreadPoolExecutor(max_workers=pool_groesse) as executor:
            aufgaben = []
            aktuell = 0

            while aktuell < iterationen:
                ende = min(aktuell + chunk_groesse, iterationen)
                aufgabe = executor.submit(self.leibniz_bereich, aktuell, ende)
                aufgaben.append(aufgabe)
                aktuell = ende

            ergebnisse = [aufgabe.result() for aufgabe in aufgaben]

        pi_viertel = sum(ergebnisse)
        pi_schaetzung = 4 * pi_viertel

        endzeit = time.time()
        laufzeit = endzeit - startzeit
        fehler = abs(pi_schaetzung - math.pi)

        messdaten = {
            'methode': 'thread_pool',
            'iterationen': iterationen,
            'laufzeit': laufzeit,
            'pi_wert': pi_schaetzung,
            'fehler': fehler,
            'arbeiter': pool_groesse
        }

        return pi_schaetzung, fehler, messdaten


class VerteilterRechner(PiBasisRechner):
    """Verteilte Berechnung über mehrere Hosts"""

    def __init__(self, port=9999):
        super().__init__()
        self.port = port

    def starte_arbeiter_server(self, host='0.0.0.0'):
        """Startet Server auf diesem Host für entfernte Berechnungen"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, self.port))
        server_socket.listen(5)

        print(f"Arbeiter-Server läuft auf {host}:{self.port}")

        while True:
            client_socket, adresse = server_socket.accept()
            try:
                # Aufgabe empfangen
                daten = client_socket.recv(4096)
                aufgabe = pickle.loads(daten)

                start, ende = aufgabe['start'], aufgabe['ende']
                ergebnis = self.leibniz_bereich(start, ende)

                # Ergebnis zurücksenden
                antwort = {'ergebnis': ergebnis}
                client_socket.send(pickle.dumps(antwort))

            except Exception as e:
                print(f"Fehler bei Aufgabenbearbeitung: {e}")
            finally:
                client_socket.close()

    def sende_aufgabe_an_host(self, host, start, ende):
        """Sendet Berechnungsaufgabe an entfernten Host"""
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((host, self.port))

            aufgabe = {'start': start, 'ende': ende}
            client_socket.send(pickle.dumps(aufgabe))

            daten = client_socket.recv(4096)
            antwort = pickle.loads(daten)

            client_socket.close()
            return antwort['ergebnis']

        except Exception as e:
            print(f"Verbindungsfehler zu {host}: {e}")
            # Fallback: lokale Berechnung
            return self.leibniz_bereich(start, ende)

    def berechne_verteilt(self, iterationen, hosts, segment_groesse=None):
        """Verteilt Berechnung auf mehrere Hosts"""
        startzeit = time.time()

        if segment_groesse is None:
            segment_groesse = iterationen // len(hosts)

        # Arbeit auf Hosts verteilen
        aufgaben = []
        aktuell = 0
        host_index = 0

        while aktuell < iterationen:
            ende = min(aktuell + segment_groesse, iterationen)
            host = hosts[host_index % len(hosts)]
            aufgaben.append((host, aktuell, ende))
            aktuell = ende
            host_index += 1

        # Aufgaben parallel ausführen
        ergebnisse = []
        with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
            futures = [executor.submit(self.sende_aufgabe_an_host, host, start, ende)
                      for host, start, ende in aufgaben]
            ergebnisse = [future.result() for future in futures]

        pi_viertel = sum(ergebnisse)
        pi_schaetzung = 4 * pi_viertel

        endzeit = time.time()
        laufzeit = endzeit - startzeit
        fehler = abs(pi_schaetzung - math.pi)

        messdaten = {
            'methode': 'verteilt',
            'iterationen': iterationen,
            'laufzeit': laufzeit,
            'pi_wert': pi_schaetzung,
            'fehler': fehler,
            'arbeiter': len(hosts),
            'hosts': hosts,
            'segment_groesse': segment_groesse
        }

        return pi_schaetzung, fehler, messdaten


def zeige_ergebnisse(pi_wert, fehler, messdaten):
    """Formatierte Ausgabe der Berechnungsergebnisse"""
    print(f"\n{'='*65}")
    print(f"π Berechnungsergebnisse - Methode: {messdaten['methode'].upper()}")
    print(f"{'='*65}")
    print(f"Iterationen:        {messdaten['iterationen']:,}")
    print(f"π Schätzung:        {pi_wert:.10f}")
    print(f"Echter π Wert:      {math.pi:.10f}")
    print(f"Absoluter Fehler:   {fehler:.2e}")
    print(f"Laufzeit:           {messdaten['laufzeit']:.4f} Sekunden")
    print(f"Arbeiter/Threads:   {messdaten['arbeiter']}")

    if 'hosts' in messdaten:
        print(f"Verwendete Hosts:   {', '.join(messdaten['hosts'])}")
    if 'segment_groesse' in messdaten:
        print(f"Segment-Größe:      {messdaten['segment_groesse']:,}")

    print(f"{'='*65}")


def main():
    """Hauptfunktion mit Kommandozeilen-Argumenten"""
    parser = argparse.ArgumentParser(
        description='Parallele π Berechnung mit Leibniz-Formel'
    )

    parser.add_argument('-i', '--iterations', type=int, default=1000000,
                        help='Anzahl Iterationen (Standard: 1000000)')
    parser.add_argument('--with-gil', action='store_true',
                        help='Berechnung mit GIL Threads')
    parser.add_argument('--with-thread', action='store_true',
                        help='Berechnung mit parallelen Threads')
    parser.add_argument('--with-proces', action='store_true',
                        help='Berechnung mit Prozessen')
    parser.add_argument('--producer-consumer', action='store_true',
                        help='Producer/Consumer Architektur verwenden')
    parser.add_argument('--map-reduce', action='store_true',
                        help='Map/Filter/Reduce verwenden')
    parser.add_argument('--pool', type=int, metavar='GROESSE',
                        help='Thread Pool mit angegebener Größe verwenden')
    parser.add_argument('--hosts', type=str,
                        help='Komma-getrennte Liste von Hosts für verteilte Berechnung')
    parser.add_argument('-s', '--start-server', type=int, metavar='PORT',
                        help='Arbeiter-Server auf angegebenem Port starten')
    parser.add_argument('--seg-size', type=int, default=10000,
                        help='Segment-Größe für verteilte Berechnung')
    parser.add_argument('--workers', type=int, default=4,
                        help='Anzahl Worker/Threads (Standard: 4)')
    parser.add_argument('--alle', action='store_true',
                        help='Alle Berechnungsmethoden zum Vergleich ausführen')

    args = parser.parse_args()

    # Server-Modus starten
    if args.start_server:
        rechner = VerteilterRechner(port=args.start_server)
        rechner.starte_arbeiter_server()
        return

    # Rechner-Instanzen erstellen
    basis_rechner = PiBasisRechner()
    gil_rechner = GILThreadRechner()
    thread_rechner = ParalleleThreadRechner()
    prozess_rechner = ProzessRechner()
    pc_rechner = ProducerConsumerRechner()
    mr_rechner = MapReduceRechner()
    pool_rechner = ThreadPoolRechner()
    verteilt_rechner = VerteilterRechner()

    alle_ergebnisse = []

    # Berechnungen basierend auf Argumenten ausführen
    if args.with_gil or args.alle:
        pi, fehler, messung = gil_rechner.berechne_mit_gil_threads(
            args.iterations, args.workers
        )
        zeige_ergebnisse(pi, fehler, messung)
        alle_ergebnisse.append(messung)

    if args.with_thread or args.alle:
        pi, fehler, messung = thread_rechner.berechne_mit_parallelen_threads(
            args.iterations, args.workers
        )
        zeige_ergebnisse(pi, fehler, messung)
        alle_ergebnisse.append(messung)

    if args.with_proces or args.alle:
        pi, fehler, messung = prozess_rechner.berechne_mit_prozessen(
            args.iterations, args.workers
        )
        zeige_ergebnisse(pi, fehler, messung)
        alle_ergebnisse.append(messung)

    if args.producer_consumer or args.alle:
        pi, fehler, messung = pc_rechner.berechne_producer_consumer(
            args.iterations, args.workers
        )
        zeige_ergebnisse(pi, fehler, messung)
        alle_ergebnisse.append(messung)

    if args.map_reduce or args.alle:
        pi, fehler, messung = mr_rechner.berechne_map_reduce(
            args.iterations, args.workers
        )
        zeige_ergebnisse(pi, fehler, messung)
        alle_ergebnisse.append(messung)

    if args.pool:
        pi, fehler, messung = pool_rechner.berechne_mit_thread_pool(
            args.iterations, args.pool
        )
        zeige_ergebnisse(pi, fehler, messung)
        alle_ergebnisse.append(messung)

    if args.hosts:
        host_liste = [host.strip() for host in args.hosts.split(',')]
        pi, fehler, messung = verteilt_rechner.berechne_verteilt(
            args.iterations, host_liste, args.seg_size
        )
        zeige_ergebnisse(pi, fehler, messung)
        alle_ergebnisse.append(messung)

    # Standard: Sequenziell + GIL wenn keine spezifische Methode gewählt
    if not any([args.with_gil, args.with_thread, args.with_proces,
                args.producer_consumer, args.map_reduce, args.pool,
                args.hosts, args.alle]):

        # Sequenzielle Referenz
        pi, fehler, messung = basis_rechner.sequenzielle_berechnung(args.iterations)
        zeige_ergebnisse(pi, fehler, messung)
        alle_ergebnisse.append(messung)

        # GIL Threads (Hauptfeature)
        pi, fehler, messung = gil_rechner.berechne_mit_gil_threads(
            args.iterations, args.workers
        )
        zeige_ergebnisse(pi, fehler, messung)
        alle_ergebnisse.append(messung)

    # Performance-Vergleich anzeigen
    if len(alle_ergebnisse) > 1:
        print(f"\n{'='*70}")
        print("PERFORMANCE-VERGLEICH")
        print(f"{'='*70}")
        print(f"{'Methode':<20} {'Zeit (s)':<12} {'Fehler':<15} {'Arbeiter':<10}")
        print(f"{'-'*70}")

        for ergebnis in alle_ergebnisse:
            print(f"{ergebnis['methode']:<20} {ergebnis['laufzeit']:<12.4f} "
                  f"{ergebnis['fehler']:<15.2e} {ergebnis['arbeiter']:<10}")

        print(f"{'='*70}")


if __name__ == "__main__":
    main()
