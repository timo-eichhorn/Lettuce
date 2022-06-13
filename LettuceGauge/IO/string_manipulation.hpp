#ifndef LETTUCE_STRING_MANIPULATION_HPP
#define LETTUCE_STRING_MANIPULATION_HPP

// Non-standard library headers
// ...
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
// ...
//----------------------------------------
// Standard C headers
// ...

// Following stuff taken from: https://www.cppstories.com/2022/ranges-perf/
// First version with temporary

std::string trimLeft(const std::string &s) {
    auto temp = s;
    temp.erase(std::begin(temp), 
                std::find_if_not(std::begin(temp), std::end(temp), isspace));
    return temp;
}

std::string trimRight(const std::string &s) {
    auto temp = s;
    temp.erase(std::find_if_not(std::rbegin(temp), std::rend(temp), isspace).base(), 
                   std::end(temp));
    return temp;
}

std::string trim(const std::string &s) {
    return trimLeft(trimRight(s));
}

// Second version without temporary

std::string trim2(const std::string &s) {
   auto wsfront=std::find_if_not(s.begin(),s.end(), isspace);
   auto wsback=std::find_if_not(s.rbegin(),s.rend(), isspace).base();
   return (wsback<=wsfront ? std::string() : std::string(wsfront,wsback));
}

std::string TrimLeft(const std::string& str)
{
    auto wsfront = std::find_if_not(str.begin(), str.end(), )
}

#endif // LETTUCE_STRING_MANIPULATION_HPP
