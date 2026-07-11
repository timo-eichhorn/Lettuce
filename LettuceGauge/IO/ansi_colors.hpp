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
#include <cstdio>
#include <unistd.h>

// Provides options to print colored output to an ANSI-compatible terminal

namespace Lettuce::Color
{
    [[nodiscard]]
    inline bool stdout_supports_colors() noexcept
    {
        return isatty(fileno(stdout)) != 0;
    }

    namespace Detail
    {
        template<typename CharT, typename Traits>
        constexpr std::basic_ostream<CharT, Traits>& Apply(std::basic_ostream<CharT, Traits>& os, const char* ansi_code)
        {
            if (stdout_supports_colors())
            {
                os << ansi_code;
            }
            return os;
        }
    } // namespace Detail

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& Reset(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[0m");
    }

    //-----
    // Colors

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& Black(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[30m");
    }

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& Red(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[31m");
    }

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& Green(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[32m");
    }

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& Yellow(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[33m");
    }

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& Blue(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[34m");
    }

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& Magenta(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[35m");
    }

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& Cyan(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[36m");
    }

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& White(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[37m");
    }

    //-----
    // Bold colors

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& BoldBlack(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[1m\033[30m");
    }

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& BoldRed(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[1m\033[31m");
    }

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& BoldGreen(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[1m\033[32m");
    }

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& BoldYellow(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[1m\033[33m");
    }

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& BoldBlue(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[1m\033[34m");
    }

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& BoldMagenta(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[1m\033[35m");
    }

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& BoldCyan(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[1m\033[36m");
    }

    template<typename CharT, typename Traits>
    constexpr std::basic_ostream<CharT, Traits>& BoldWhite(std::basic_ostream<CharT, Traits>& os)
    {
        return Detail::Apply(os, "\033[1m\033[37m");
    }
} // namespace Lettuce::Color

#endif // LETTUCE_COLORS_HPP
