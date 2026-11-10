"""Rasgos léxicos de los modelos anteriores del curso (Python puro).

El léxico ``LEXICO_ES`` proviene de ``examples/logic/lexical_features.py`` del repositorio
del curso (github.com/EdwinPuertas/courseNLP), usado en ``FeatureExtraction`` para TASS.
Aquí se reimplementan los rasgos para reseñas, con cuatro correcciones documentadas
respecto a ``feature_extraction.py``:

1. Léxico y texto se normalizan igual (minúsculas, sin tildes): en el original las
   entradas con tilde ("él", "aún", "más"...) no se alcanzaban tras quitar acentos.
2. Curtosis y asimetría conservan el signo (el original aplicaba ``abs()`` a todo el vector).
3. La diversidad léxica es tipos/tokens de *palabras* (el original contaba caracteres distintos).
4. Se omiten ``weighted_position``/``weighted_normalized`` (dependen casi solo de la longitud)
   y las etiquetas de Twitter (mention, url, hashtag, emoji, rt), que no aplican a reseñas.

Se usa igual en entrenamiento (``ml/lexical.py``) y en producción (``core/model.py``),
lo que garantiza que los rasgos sean idénticos en ambos lados.
"""
from __future__ import annotations

import re

from .text import normalizar

LEXICO_ES = {'first_person_singular': ['yo', 'mio', 'mi'],
 'second_person_singular': ['tú', 'tu', 'tuyo', 'usted', 'vos'],
 'third_person_singular': ['él', 'el', 'ella', 'ese', 'esa', 'aquel'],
 'first_person_plurar': ['nosotros', 'nuestros'],
 'second_person_plurar': ['tuyos', 'tuyas', 'ustedes', 'vosotros', 'vosotras'],
 'third_person_plurar': ['ellos', 'ellas', 'ese', 'esos', 'aquellas', 'aquellos'],
 'adverb_time': ['ya',
                 'aún',
                 'hoy',
                 'tarde',
                 'pronto',
                 'todavía',
                 'ayer',
                 'recién',
                 'nunca',
                 'siempre',
                 'jamás',
                 'ahora'],
 'adverb_neg': ['tampoco', 'no', 'en absoluto', 'nunca'],
 'adverb_place': ['ahí',
                  'allí',
                  'aquí',
                  'acá',
                  'delante',
                  'detrás',
                  'arriba',
                  'abajo',
                  'cerca',
                  'lejos',
                  'encima',
                  'fuera',
                  'dentro'],
 'adverb_mode': ['mal',
                 'bien',
                 'regular',
                 'despacio',
                 'así',
                 'mejor',
                 'peor',
                 'claro',
                 'similar',
                 'fácilmente'],
 'adverb_cant': ['muy', 'más', 'poco', 'bastante', 'bastante', 'menos', 'mucho', 'algo', 'casi'],
 'adjetives_neg': ['aburrido',
                   'ácido',
                   'amargo',
                   'ancho',
                   'atrevido',
                   'bajo',
                   'blando',
                   'caliente',
                   'común',
                   'conocido',
                   'corto',
                   'débil',
                   'delgado',
                   'diferente',
                   'difícil',
                   'duro',
                   'enfermo',
                   'estrecho',
                   'falso',
                   'feo',
                   'frío',
                   'húmedo',
                   'imposible',
                   'izquierdo',
                   'lento',
                   'malo',
                   'menor',
                   'muerto',
                   'negro',
                   'peor',
                   'pequeño',
                   'pobre',
                   'poco',
                   'popular',
                   'raro',
                   'rojo',
                   'salado',
                   'simple',
                   'sinvergüenza',
                   'solo',
                   'soso',
                   'tímido',
                   'tonto',
                   'triste',
                   'viejo'],
 'adjetives_pos': ['alegre',
                   'alto',
                   'azul',
                   'blanco',
                   'bonito',
                   'buen',
                   'bueno',
                   'caliente',
                   'capaz',
                   'central',
                   'contento',
                   'derecho',
                   'diferente',
                   'divertido',
                   'dulce',
                   'duro',
                   'exterior',
                   'famoso',
                   'final',
                   'fresco',
                   'fuerte',
                   'gran',
                   'grande',
                   'guapo',
                   'guay',
                   'igual',
                   'imposible',
                   'interesante',
                   'joven',
                   'largo',
                   'listo',
                   'mayor',
                   'mejor',
                   'mucho',
                   'musical',
                   'nacional',
                   'natural',
                   'nuevo',
                   'perfecto',
                   'popular',
                   'posible',
                   'primer',
                   'primero',
                   'principal',
                   'próximo',
                   'rápido',
                   'real',
                   'recto',
                   'rico',
                   'salado',
                   'sano',
                   'seco',
                   'útil',
                   'verdadero',
                   'verde',
                   'vivo'],
 'who_general': ['familia',
                 'familiar',
                 'familiares',
                 'acompañante',
                 'aprendices',
                 'aprendiz',
                 'artista',
                 'asistente',
                 'compinche',
                 'directiva',
                 'directivo',
                 'especialista',
                 'estudiante',
                 'estudiantes',
                 'joven',
                 'jovenes',
                 'lider',
                 'pareja',
                 'projimo'],
 'who_male': ['abuelo',
              'alcalde',
              'medico',
              'alumno',
              'alumnos',
              'amigo',
              'amigos',
              'asesor',
              'chiquillo',
              'compa',
              'compadre',
              'compañero',
              'compañeros',
              'decano',
              'director',
              'doctor',
              'empleado',
              'empresario',
              'entrenador',
              'esposo',
              'esposito',
              'experto',
              'expertos',
              'funcionario',
              'hermano',
              'ingeniero',
              'investigador',
              'jefe',
              'jugador',
              'padre',
              'marido',
              'medico',
              'muchacho',
              'muchachos',
              'niño',
              'niños',
              'novio',
              'hijo',
              'primo',
              'papa',
              'pequeños',
              'profesor',
              'pequeño',
              'secretario',
              'socio',
              'tutor',
              'papi',
              'tio',
              'nieto'],
 'who_female': ['alumna',
                'alumnas',
                'amiga',
                'amigas',
                'asesora',
                'chiquilla',
                'comadre',
                'compañera',
                'compañeras',
                'dama',
                'decana',
                'directora',
                'doctora',
                'empleada',
                'empresaria',
                'esposa',
                'esposita',
                'experta',
                'expertas',
                'funcionaria',
                'hermana',
                'ingeniera',
                'investigadora',
                'jefa',
                'jugadora',
                'madre',
                'mama',
                'medica',
                'muchacha',
                'muchachas',
                'mujer',
                'niñas',
                'novia',
                'novicia',
                'hija',
                'prima',
                'pequeña',
                'pequeñas',
                'profesora',
                'secretaria',
                'señora',
                'socia',
                'tutora',
                'mami',
                'tia',
                'nieta'],
 'hate': ['animosidad',
          'antagonismo',
          'enemistad',
          'odio',
          'horror',
          'hostilidad',
          'aborrecimiento',
          'dolor',
          'rencor',
          'resentimiento',
          'venganza',
          'veneno',
          'aborrecimiento',
          'abominación ',
          'anatema',
          'animus',
          'antipatía',
          'aversión',
          'molestar',
          'bugbear',
          'detestación ',
          'disgusto',
          'execración',
          'escarcha',
          'agravio',
          'queja',
          'irritante',
          'malevolencia',
          'malignidad',
          'molestia',
          'objeción',
          'odio',
          'irritación',
          'repugnancia',
          'repulsión',
          'repulsión',
          'desprecio',
          'despecho',
          'problemas',
          'bestia negra',
          'mala voluntad',
          'desagradable',
          'mirada desagradable',
          'ningún amor perdido']}

_PALABRA = re.compile(r"\w+")


def _preparar(lexico: dict) -> dict:
    out = {}
    for cat, entradas in lexico.items():
        unas, frases = set(), []
        for e in entradas:
            n = normalizar(e)
            if not n:
                continue
            if " " in n:
                frases.append(re.compile(r"\b" + re.escape(n) + r"\b"))
            else:
                unas.add(n)
        out[cat] = (unas, frases)
    return out


_LEX = _preparar(LEXICO_ES)
CATEGORIAS = list(LEXICO_ES.keys())
ADVERBIOS = ["adverb_time", "adverb_neg", "adverb_place", "adverb_mode", "adverb_cant"]
NOMBRES = CATEGORIAS + ["adverb_all", "n_words", "avg_word", "kur_word", "skew_word", "lexical_diversity"]


def _momentos(xs: list[float]) -> tuple[float, float, float]:
    """Media, asimetría y curtosis (Fisher), sesgadas, como scipy.stats por defecto."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = sum(xs) / n
    m2 = sum((x - m) ** 2 for x in xs) / n
    if m2 == 0:
        return m, 0.0, 0.0
    m3 = sum((x - m) ** 3 for x in xs) / n
    m4 = sum((x - m) ** 4 for x in xs) / n
    return m, m3 / m2 ** 1.5, m4 / m2 ** 2 - 3.0


def rasgos(texto: str) -> list[float]:
    """Vector de rasgos léxicos, en el orden de ``NOMBRES``."""
    t = normalizar(texto)
    tokens = _PALABRA.findall(t)
    vec = {}
    for cat in CATEGORIAS:
        unas, frases = _LEX[cat]
        vec[cat] = float(sum(1 for w in tokens if w in unas) + sum(len(p.findall(t)) for p in frases))
    vec["adverb_all"] = float(sum(vec[a] for a in ADVERBIOS))
    largos = [float(len(w)) for w in tokens]
    media, asim, curt = _momentos(largos)
    vec["n_words"] = float(len(tokens))
    vec["avg_word"] = media
    vec["kur_word"] = curt
    vec["skew_word"] = asim
    vec["lexical_diversity"] = len(set(tokens)) / len(tokens) if tokens else 0.0
    return [vec[n] for n in NOMBRES]
