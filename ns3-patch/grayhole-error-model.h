/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Grayhole attack error model for LEO satellite networks.
 *
 * Emulates a malicious satellite that selectively drops packets instead of
 * forwarding them. Supports three attack patterns:
 *   CONSTANT : drop with probability DropRate from StartTime onward
 *   ON_OFF   : alternate active (OnDuration) / benign (OffDuration) windows
 *   SCAN     : alias of ON_OFF (periodic attack scanning)
 */
#ifndef GRAYHOLE_ERROR_MODEL_H
#define GRAYHOLE_ERROR_MODEL_H

#include <string>
#include "ns3/error-model.h"
#include "ns3/nstime.h"
#include "ns3/random-variable-stream.h"

namespace ns3 {

class GrayholeErrorModel : public ErrorModel
{
public:
  enum Mode
  {
    CONSTANT,
    ON_OFF,
    SCAN
  };

  static TypeId GetTypeId (void);
  static Mode ParseMode (const std::string &modeStr);

  GrayholeErrorModel ();
  virtual ~GrayholeErrorModel ();

  void SetMode (Mode mode);
  Mode GetMode (void) const;

  void SetDropRate (double rate);
  double GetDropRate (void) const;

  void SetStartTime (Time startTime);
  Time GetStartTime (void) const;

  void SetEndTime (Time endTime);   // Time(0) => until simulation end
  Time GetEndTime (void) const;

  void SetOnDuration (Time onDuration);
  void SetOffDuration (Time offDuration);

private:
  virtual bool DoCorrupt (Ptr<Packet> p);
  virtual void DoReset (void);

  bool IsAttackActive (void) const;

  Mode m_mode;
  double m_dropRate;
  Time m_startTime;
  Time m_endTime;
  Time m_onDuration;
  Time m_offDuration;
  Ptr<UniformRandomVariable> m_random;
};

} // namespace ns3

#endif // GRAYHOLE_ERROR_MODEL_H