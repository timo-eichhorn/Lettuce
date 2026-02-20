#ifndef LETTUCE_TIMER_HPP
#define LETTUCE_TIMER_HPP

// Non-standard library headers
// ...
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <chrono>
#include <ratio>
//----------------------------------------
// Standard C headers
// ...

template<typename ClockT = std::chrono::steady_clock>
class Timer
{
private:
    using TimePointT = typename ClockT::time_point;
    TimePointT start_time;
public:
    Timer() noexcept(noexcept(ClockT::now())) : start_time(ClockT::now()) {}

    void Reset()
    {
        start_time = ClockT::now();
    }

    template<typename PeriodT = std::milli>
    [[nodiscard]]
    auto GetTime() const noexcept(noexcept(ClockT::now()))
    {
        using FloatDurationT = std::chrono::duration<double, PeriodT>;
        return std::chrono::duration_cast<FloatDurationT>(ClockT::now() - start_time).count();
    }

    [[nodiscard]]
    auto GetTimeSeconds() const noexcept(noexcept(ClockT::now()))
    {
        return GetTime<std::ratio<1, 1>>();
    }
};

#endif // LETTUCE_TIMER_HPP
