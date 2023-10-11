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

std::size_t FindTokenEnd(const std::string_view str, const std::string_view token)
{
    // First check for leading whitespaces (essentially std::isspace)
    std::size_t first_nonwhitespace {str.find_first_not_of(" \f\n\r\t\v")};
    if (first_nonwhitespace == std::string::npos)
    {
        return std::string::npos;
    }
    // Ignoring leading whitespaces, check if the remaining string starts with 'token'
    // TODO: What about trailing whitespaces?
    if (str.substr(first_nonwhitespace).starts_with(token))
    {
        return first_nonwhitespace + token.length();
    }
    else
    {
        return std::string::npos;
    }
}

// C++ strings suck, so define this helper function
void LeftErase(std::string& str, const std::string& erase)
{
    std::size_t pos = str.find(erase);
    if (pos != std::string::npos)
    {
        str.erase(pos, erase.length());
    }
}

// Search the string 'str' for 'erase' starting and delete everything to the left of 'erase'
// If including == true, erase everything including 'erase', otherwise only until 'erase'
void EraseUntil(std::string& str, const std::string& erase, const bool including = true)
{
    std::size_t pos = str.find(erase);
    if (pos != std::string::npos)
    {
        if (including)
        {
            str.erase(0, pos + erase.length());
        }
        else
        {
            str.erase(0, pos);
        }
    }
}

template<typename T>
T ConvertStringTo(const std::string& str)
{
    std::istringstream ss(str);
    T result;
    ss >> result;
    return result;
}

// Following stuff taken from: https://www.cppstories.com/2022/ranges-perf/
// First version with temporary
// TODO: Currently never included anywhere

// std::string trimLeft(const std::string &str)
// {
//     auto tmp = str;
//     tmp.erase(std::begin(tmp), std::find_if_not(std::begin(tmp), std::end(tmp), isspace));
//     return tmp;
// }

// std::string trimRight(const std::string &str)
// {
//     auto tmp = str;
//     tmp.erase(std::find_if_not(std::rbegin(tmp), std::rend(tmp), isspace).base(), std::end(tmp));
//     return tmp;
// }

// std::string trim(const std::string &str)
// {
//     return trimLeft(trimRight(str));
// }

// // Second version without temporary

// std::string trim2(const std::string &str)
// {
//     auto wsfront = std::find_if_not(str.begin(), str.end(), isspace);
//     auto wsback  = std::find_if_not(str.rbegin(), str.rend(), isspace).base();
//     return (wsback <= wsfront ? std::string() : std::string(wsfront, wsback));
// }

// std::string TrimLeft(const std::string& str)
// {
//     auto wsfront = std::find_if_not(str.begin(), str.end(), )
// }

#endif // LETTUCE_STRING_MANIPULATION_HPP
