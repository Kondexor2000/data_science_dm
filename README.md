# Komentarze IG: klasyfikacja intencji i tonu

Portfolio Data Science / NLP w języku polskim. Projekt rozpoznaje kategorię komentarza pod postem na Instagramie:

- **pozytywny** — wypowiedź niezwiązana z wyglądem, np. odnosząca się do twórczości, osiągnięć lub zainteresowań;
- **neutralny** — komentarz dotyczący wyglądu, bez jawnej wrogości;
- **negatywny** — obraźliwy, poniżający, napastliwy lub seksualizujący komentarz.

To etykiety operacyjne do moderacji i analizy, a nie ocena osoby ani diagnoza intencji. Kontekst może zmienić znaczenie wypowiedzi. W szczególności kategoria neutralna nie oznacza automatycznie, że komentarz jest mile widziany przez odbiorczynię.

## MVP

1. Wczytanie ręcznie oznaczonego pliku CSV (`text,label`) i walidacja schematu.
2. Podział train/test ze stratyfikacją, pipeline TF-IDF + regresja logistyczna.
3. Raport precision, recall, F1 i macierzy pomyłek dla każdej klasy; zapis modelu i metryk.
4. REST API z `/health` i `/predict` zwracające kategorię oraz prawdopodobieństwa.

Zakres MVP celowo nie obejmuje pobierania komentarzy z Instagrama ani automatycznego ukrywania treści. Najpierw potrzebny jest legalnie pozyskany, reprezentatywny i zgodnie oznaczony zbiór danych. `data/sample.csv` to mały, syntetyczny przykład do demonstracji działania — nie nadaje się do oceny jakości ani użycia produkcyjnego.

## Co jest potrzebne, a co opcjonalne

Do podstawowego uruchomienia projektu wystarczą Python, zależności z `requirements.txt` oraz lokalne środowisko wirtualne. Na tej ścieżce można wytrenować model i uruchomić API — Docker i Jenkins nie są wymagane.

Docker jest opcjonalnym sposobem na uruchomienie API w kontenerze. Jenkins jest przykładowym pipeline'em CI, który pokazuje automatyczną kontrolę projektu i budowanie obrazu; nie jest potrzebny do treningu ani lokalnego uruchomienia. Można pominąć obie sekcje, jeśli celem jest tylko praca z modelem i API.

## Uruchomienie lokalne

Wymagania: Python 3.11+. W PowerShell, z katalogu projektu:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m ig_comments.train --data data/sample.csv --output artifacts/model.joblib
python -m uvicorn ig_comments.api:app --host 0.0.0.0 --port 8000
```

Jeśli używasz już aktywnego środowiska wirtualnego, pomiń tworzenie i aktywację `.venv`. Instaluj pakiety przez `python -m pip`, aby trafiły do tego samego interpretera, którym uruchamiasz moduł.

API: `POST /predict` z JSON-em `{"text":"Świetny montaż, dużo się nauczyłem!"}`. Interaktywny opis endpointów: `/docs`.

## Docker (opcjonalnie)

Docker nie jest potrzebny do lokalnego treningu ani uruchomienia API. Jeśli chcesz sprawdzić wariant kontenerowy, z katalogu projektu wykonaj:

```bash
docker build -t ig-comments:local .
docker run --rm -p 8000:8000 ig-comments:local
```

API będzie dostępne pod `http://localhost:8000`, a dokumentacja pod `http://localhost:8000/docs`. Jeśli port 8000 jest już zajęty, można użyć innego portu na komputerze, np. `docker run --rm -p 8001:8000 ig-comments:local` — wtedy adres API to `http://localhost:8001`.

Obraz trenuje demonstracyjny model podczas budowania. W rzeczywistym wdrożeniu model należy trenować w kontrolowanym procesie na zatwierdzonym zbiorze, wersjonować artefakt i wdrażać obraz z przypiętym modelem.

## Jenkins (opcjonalnie, demonstracja CI)

`Jenkinsfile` definiuje przykładowy pipeline: checkout, instalację zależności, kontrolę składni i budowę obrazu Docker. Do jego uruchomienia potrzebny jest skonfigurowany Jenkins oraz agent z Pythonem 3.11+ i dostępem do Docker CLI/daemon. Pipeline nie wdraża aplikacji ani nie publikuje obrazu. Jeśli nie używasz Jenkinsa, możesz pominąć tę część.

## Struktura

```text
data/sample.csv             demonstracyjne, syntetyczne dane
src/ig_comments/train.py    walidacja, uczenie, metryki i zapis modelu
src/ig_comments/api.py      API predykcji FastAPI
Dockerfile                  obraz uruchomieniowy
Jenkinsfile                 CI dla checkoutu, składni i budowy obrazu
```

## Decyzje i kryteria jakości

- **Baseline:** TF-IDF słów i bigramów + LogisticRegression; jest szybki, tani i łatwy do interpretacji jako punkt odniesienia.
- **Metryki:** raportujemy per-class precision/recall/F1 oraz macro-F1. Szczególnie monitorujemy recall klasy negatywnej, ale przy niskiej precyzji system mógłby błędnie flagować zwykłe wypowiedzi.
- **Próg wdrożenia:** ustalany po ewaluacji na niezależnym, reprezentatywnym zbiorze oraz przeglądzie błędów przez ludzi. Nie ma uczciwego progu liczbowego przed zebraniem takich danych.
- **Ochrona danych:** minimalizować dane, usuwać identyfikatory, ograniczać retencję, nie trenować na prywatnych wiadomościach bez odpowiedniej podstawy i zgody. Publiczny komentarz również może zawierać dane osobowe.
- **Ograniczenia:** ironia, cytowanie obelg, slang, dialekty, kontekst zdjęcia i relacja między rozmówcami mogą prowadzić do błędów. Model wspiera moderatora; nie powinien samodzielnie karać użytkowników.

## Rozszerzenia po MVP

Zebrać i opisać politykę anotacji, mierzyć zgodność anotatorów, dodać grupową analizę błędów i testy dryfu, wersjonowanie danych/modelu (np. DVC/MLflow), monitoring oraz przegląd człowieka. Rozważyć czwartą klasę „niejednoznaczne” albo multilabel, jeżeli jeden komentarz może jednocześnie chwalić i oceniać wygląd. Dopiero potem porównać baseline z polskim modelem transformerowym.
