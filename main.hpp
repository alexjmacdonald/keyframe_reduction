#define VIDEO_ROWS      322
#define VIDEO_COLS      240
#define VIDEO_GRIDSIZE  5

#include <cstdint>

template<typename Ordinal, typename Iterator>
bool seek(bool use_container, Iterator &it, Iterator &end, Ordinal &frame_idx, Ordinal &frame_cap);

uint8_t grayscale(const uint8_t r, const uint8_t g, const uint8_t b);

template <uint32_t GRID, uint32_t _ROW_WIDTH, uint32_t _COL_WIDTH>
uint32_t maploc(uint32_t r, uint32_t c) {
  // TODO validation strategy,
  // right now we assume the caller isn't going to send OOB coords
  // this is probably appropriate (avoid branching) but should circle back

  // If the grid size doesn't evenly divide,
  // we simply ignore the margin.
  constexpr auto ROW_WIDTH = _ROW_WIDTH - (_ROW_WIDTH % GRID);
  constexpr auto COL_WIDTH = _COL_WIDTH - (_COL_WIDTH % GRID);
  // Consider a 3x3 grid over 9x9 pixels:
  /*
    * a b c  x x x  x x x
    * d e f  x x x  x x x
    * g h i  x x x  x x x
    *
    * x x x  x x x  x x x
    * x x x  x x x  x x x
    * x x x  x x x  x x x
    *
    * x x x  x x x  x x x
    * x x x  x x x  x x x
    * x x x  x x x  x x x
    */
  // For a regular linear traversal:
  // [ a b c . . . . . . d e f ... ]
  // the members of a cell aren't consecutive.
  //
  // For efficiency's sake (maintaining cache coherency & avoiding needless reiteration)
  // we make one pass across the image, copying pixels to a new buffer, and mapping
  // them to new locations in the buffer so that we get:
  // [ a b c d e f g h i . . . . . . ... ]
  //
  // The formula for this looks like:
  //
  /*
    *  (using ordinal ranks, not 0-based indexing)
    *
    *  Presume we map the contents of each grid cell
    *  to its grid's new index in the new buffer:
    *
    *    x- - - - - - - - - - - - - -
    *  y
    *  |  1  1  1   2  2  2   3  3  3
    *  |  1  1  1   2  2  2   3  3  3
    *  |  1  1  1   2  2  2   3  3  3
    *
    *  |  4  4  4   5  5  5   6  6  6
    *  |  4  4  4   5  5  5   6  6  6
    *  |  4  4  4   5  5  5   6  6  6
    *
    *  and then we account for the grid size
    *  (so, 1+0, 1+9, 1+9+9...)
    *  so that each cell maps to the grid's actual mem index:
    *
    *    x- - - - - - - - - - - - - -
    *  y
    *  |  1  1  1   10 10 10  19 19 19
    *     1  1  1   10 10 10  19 19 19
    *  |  1  1  1   10 10 10  19 19 19
    *
    *  |  28 28 28  37 37 37  46 46 46
    *     28 28 28  37 37 37  46 46 46
    *  |  28 28 28  37 37 37  46 46 46
    *
    *  Pretty close!
    *
    *  Finally, we use addition to compose our grid cell's index
    *  with the cell element's index within the grid cell:
    *
    *    x- - - - - - - - - - - - - -
    *  y
    *  |  0  1  2   0  1  2   0  1  2
    *     3  4  5   3  4  5   3  4  5
    *  |  6  7  8   6  7  8   6  7  8
    *
    *  |  0  1  2   0  1  2   0  1  2
    *     3  4  5   3  4  5   3  4  5
    *  |  6  7  8   6  7  8   6  7  8
    *
    *  Regular piecewise matrix addition.
    *
    *  Therefore...
    *
    *    x- - - - - - - - - - - - - -
    *  y
    *  |  1  2  3   4  5  6   7  8  9
    *     10 11 12  13 14 15  16 17 18
    *  |  19 20 21  22 23 24  25 26 27
    *
    *  |  28 29 30  31 32 33  34 35 36
    *     37 38 39  40 41 42  43 44 45
    *  |  46 47 48  49 50 51  52 53 54
    *
    *     becomes
    *
    *    x- - - - - - - - - - - - - -
    *  y
    *  |  1  2  3   10 11 12  19 20 21
    *     4  5  6   13 14 15  22 23 24
    *  |  7  8  9   16 17 18  25 26 27
    *
    *  |  28 29 30  37 38 39  46 47 48
    *     31 32 33  40 41 42  49 50 51
    *  |  34 35 36  43 44 45  52 53 54
    */

    const uint32_t cells_above        = (r/GRID) * (COL_WIDTH / GRID);
    const uint32_t cells_before       = c / GRID;
    const uint32_t starting_grid_idx  = (GRID*GRID) * (cells_above + cells_before);
    const uint32_t within_grid_idx    = (r%GRID)*GRID + c%GRID;

    return (starting_grid_idx + within_grid_idx);
}

template <uint32_t ROWS, uint32_t COLS, uint32_t GRIDSIZE, typename ResultContainer, typename PathCollection, typename FrameIndexIterator, typename FrameIndexSequence>
bool reduce_frames(ResultContainer &result, PathCollection &paths, uint32_t &frame_idx, FrameIndexSequence &frame_indices, FrameIndexIterator &frame_it, FrameIndexIterator &frame_it_end);
