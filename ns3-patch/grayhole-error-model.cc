/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#include "grayhole-error-model.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/abort.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("GrayholeErrorModel");
NS_OBJECT_ENSURE_REGISTERED (GrayholeErrorModel);

TypeId
GrayholeErrorModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::GrayholeErrorModel")
    .SetParent<ErrorModel> ()
    .SetGroupName ("SatelliteNetwork")
    .AddConstructor<GrayholeErrorModel> ()
    .AddAttribute ("DropRate",
                   "Probability of dropping a packet while the attack is active.",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&GrayholeErrorModel::m_dropRate),
                   MakeDoubleChecker<double> (0.0, 1.0))
    .AddAttribute ("StartTime",
                   "Simulation time at which the attack starts.",
                   TimeValue (Seconds (0)),
                   MakeTimeAccessor (&GrayholeErrorModel::m_startTime),
                   MakeTimeChecker ())
    .AddAttribute ("EndTime",
                   "Simulation time at which the attack ends (0 = until end).",
                   TimeValue (Seconds (0)),
                   MakeTimeAccessor (&GrayholeErrorModel::m_endTime),
                   MakeTimeChecker ())
    .AddAttribute ("OnDuration",
                   "Length of the active window (ON_OFF/SCAN mode).",
                   TimeValue (Seconds (1)),
                   MakeTimeAccessor (&GrayholeErrorModel::m_onDuration),
                   MakeTimeChecker ())
    .AddAttribute ("OffDuration",
                   "Length of the benign window (ON_OFF/SCAN mode).",
                   TimeValue (Seconds (1)),
                   MakeTimeAccessor (&GrayholeErrorModel::m_offDuration),
                   MakeTimeChecker ())
    ;
  return tid;
}

GrayholeErrorModel::Mode
GrayholeErrorModel::ParseMode (const std::string &modeStr)
{
  if (modeStr == "ON_OFF" || modeStr == "on_off")
    {
      return ON_OFF;
    }
  if (modeStr == "SCAN" || modeStr == "scan")
    {
      return SCAN;
    }
  return CONSTANT;
}

GrayholeErrorModel::GrayholeErrorModel ()
  : m_mode (CONSTANT),
    m_dropRate (0.0),
    m_startTime (Seconds (0)),
    m_endTime (Seconds (0)),
    m_onDuration (Seconds (1)),
    m_offDuration (Seconds (1)),
    m_random (CreateObject<UniformRandomVariable> ())
{
}

GrayholeErrorModel::~GrayholeErrorModel ()
{
}

void
GrayholeErrorModel::SetMode (Mode mode)
{
  m_mode = mode;
}

GrayholeErrorModel::Mode
GrayholeErrorModel::GetMode (void) const
{
  return m_mode;
}

void
GrayholeErrorModel::SetDropRate (double rate)
{
  NS_ABORT_MSG_UNLESS (rate >= 0.0 && rate <= 1.0, "Drop rate must be in [0,1]");
  m_dropRate = rate;
}

double
GrayholeErrorModel::GetDropRate (void) const
{
  return m_dropRate;
}

void
GrayholeErrorModel::SetStartTime (Time startTime)
{
  m_startTime = startTime;
}

Time
GrayholeErrorModel::GetStartTime (void) const
{
  return m_startTime;
}

void
GrayholeErrorModel::SetEndTime (Time endTime)
{
  m_endTime = endTime;
}

Time
GrayholeErrorModel::GetEndTime (void) const
{
  return m_endTime;
}

void
GrayholeErrorModel::SetOnDuration (Time onDuration)
{
  m_onDuration = onDuration;
}

void
GrayholeErrorModel::SetOffDuration (Time offDuration)
{
  m_offDuration = offDuration;
}

bool
GrayholeErrorModel::IsAttackActive (void) const
{
  Time now = Simulator::Now ();
  if (now < m_startTime)
    {
      return false;
    }
  if (m_endTime > Time (0) && now >= m_endTime)
    {
      return false;
    }

  switch (m_mode)
    {
    case CONSTANT:
      return true;

    case ON_OFF:
    case SCAN:
      {
        int64_t period = (m_onDuration + m_offDuration).GetNanoSeconds ();
        if (period <= 0)
          {
            return true;
          }
        int64_t elapsed = (now - m_startTime).GetNanoSeconds ();
        int64_t phase = elapsed % period;
        return phase < m_onDuration.GetNanoSeconds ();
      }

    default:
      return false;
    }
}

bool
GrayholeErrorModel::DoCorrupt (Ptr<Packet> p)
{
  if (!IsAttackActive ())
    {
      return false;
    }
  return m_random->GetValue () < m_dropRate;
}

void
GrayholeErrorModel::DoReset (void)
{
  // No internal state beyond the RNG; mirrors RateErrorModel.
}

} // namespace ns3