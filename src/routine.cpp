
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Routine base class (see the header for information about the class).
//
// =================================================================================================

#include <string>
#include <vector>

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// Constructor: not much here, because no status codes can be returned
Routine::Routine(Queue &queue, EventPointer event, const std::string &name,
                 const std::vector<std::string> &routines, const Precision precision,
                 const std::vector<Database::DatabaseEntry> &userDatabase):
    precision_(precision),
    routine_name_(name),
    queue_(queue),
    event_(event),
    context_(queue_.GetContext()),
    device_(queue_.GetDevice()),
    device_name_(device_.Name()),
    db_(queue_, routines, precision_, userDatabase) {
}

// =================================================================================================
} // namespace clblast
