/// @file library.hpp
/// @ingroup group_image
#pragma once

namespace numkit {

/// @addtogroup group_image
/// @{
class PluginContext;

class ImageLibrary
{
public:
    static void install(PluginContext &engine);
};

/// @}
} // namespace numkit
