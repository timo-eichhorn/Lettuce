#ifndef LETTUCE_COLORS_HPP
#define LETTUCE_COLORS_HPP

// Non-standard library headers
// ...
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <ostream>
//----------------------------------------
// Standard C headers
#include <unistd.h>

// Provides options to print colored output to an ANSI compatible terminal

namespace Lettuce::Color
{
    bool stdout_supports_colors()
    {
        return isatty(fileno(stdout));
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& Reset(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[0m";
        }
        else
        {
            return os;
        }
    }

    //-----
    // Colors

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& Black(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[0m";
        }
        else
        {
            return os;
        }
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& Red(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[31m";
        }
        else
        {
            return os;
        }
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& Green(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[32m";
        }
        else
        {
            return os;
        }
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& Yellow(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[33m";
        }
        else
        {
            return os;
        }
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& Blue(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[34m";
        }
        else
        {
            return os;
        }
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& Magenta(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[35m";
        }
        else
        {
            return os;
        }
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& Cyan(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[36m";
        }
        else
        {
            return os;
        }
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& White(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[37m";
        }
        else
        {
            return os;
        }
    }

    //-----
    // Bold colors

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& BoldBlack(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[1m\033[30m";
        }
        else
        {
            return os;
        }
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& BoldRed(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[1m\033[31m";
        }
        else
        {
            return os;
        }
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& BoldGreen(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[1m\033[32m";
        }
        else
        {
            return os;
        }
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& BoldYellow(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[1m\033[33m";
        }
        else
        {
            return os;
        }
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& BoldBlue(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[1m\033[34m";
        }
        else
        {
            return os;
        }
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& BoldMagenta(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[1m\033[35m";
        }
        else
        {
            return os;
        }
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& BoldCyan(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[1m\033[36m";
        }
        else
        {
            return os;
        }
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& BoldWhite(std::basic_ostream<CharT, Traits>& os)
    {
        if (stdout_supports_colors())
        {
            return os << "\033[1m\033[37m";
        }
        else
        {
            return os;
        }
    }
} // namespace Lettuce::Color

#endif // LETTUCE_COLORS_HPP
