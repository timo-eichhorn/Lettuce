#ifndef LETTUCE_STRING_MANIPULATION_HPP
#define LETTUCE_STRING_MANIPULATION_HPP

// Non-standard library headers
// ...
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <algorithm>
#include <string>
//----------------------------------------
// Standard C headers
// ...

// Following stuff taken from: https://www.cppstories.com/2022/ranges-perf/
// First version with temporary
// TODO: Currently never included anywhere

std::string trimLeft(const std::string &str)
{
    auto tmp = str;
    tmp.erase(std::begin(tmp), std::find_if_not(std::begin(tmp), std::end(tmp), isspace));
    return tmp;
}

std::string trimRight(const std::string &str)
{
    auto tmp = str;
    tmp.erase(std::find_if_not(std::rbegin(tmp), std::rend(tmp), isspace).base(), std::end(tmp));
    return tmp;
}

std::string trim(const std::string &str)
{
    return trimLeft(trimRight(str));
}

// Second version without temporary

std::string trim2(const std::string &str)
{
    auto wsfront = std::find_if_not(str.begin(), str.end(), isspace);
    auto wsback  = std::find_if_not(str.rbegin(), str.rend(), isspace).base();
    return (wsback <= wsfront ? std::string() : std::string(wsfront, wsback));
}

std::string TrimLeft(const std::string& str)
{
    auto wsfront = std::find_if_not(str.begin(), str.end(), )
}

#endif // LETTUCE_STRING_MANIPULATION_HPP
