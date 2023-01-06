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
// ...

// Provides options to print colored output to an ANSI compatible terminal

namespace let::col
{
    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& reset(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[0m";
    }

    //-----
    // Colors

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& black(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[30m";
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& red(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[31m";
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& green(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[32m";
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& yellow(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[33m";
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& blue(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[34m";
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& magenta(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[35m";
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& cyan(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[36m";
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& white(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[37m";
    }

    //-----
    // Bold colors

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& boldblack(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[1m\033[30m";
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& boldred(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[1m\033[31m";
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& boldgreen(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[1m\033[32m";
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& boldyellow(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[1m\033[33m";
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& boldblue(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[1m\033[34m";
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& boldmagenta(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[1m\033[35m";
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& boldcyan(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[1m\033[36m";
    }

    template<typename CharT, typename Traits>
    constexpr
    std::basic_ostream<CharT, Traits>& boldwhite(std::basic_ostream<CharT, Traits>& os)
    {
        return os << "\033[1m\033[37m";
    }
} // namespace let::col

#endif // LETTUCE_COLORS_HPP
