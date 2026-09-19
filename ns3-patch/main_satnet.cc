/*
 * Copyright (c) 2020 ETH Zurich
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Simon               2020
 */

#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <chrono>
#include <stdexcept>
#include <vector>
#include <sstream>

#include "ns3/basic-simulation.h"
#include "ns3/tcp-flow-scheduler.h"
#include "ns3/udp-burst-scheduler.h"
#include "ns3/pingmesh-scheduler.h"
#include "ns3/topology-satellite-network.h"
#include "ns3/tcp-optimizer.h"
#include "ns3/arbiter-single-forward-helper.h"
#include "ns3/ipv4-arbiter-routing-helper.h"
#include "ns3/gsl-if-bandwidth-helper.h"
#include "ns3/grayhole-error-model.h"
#include "ns3/point-to-point-laser-net-device.h"
#include "ns3/gsl-net-device.h"

using namespace ns3;

static std::vector<int64_t>
ParseSatelliteIds(const std::string& csv) {
    std::vector<int64_t> ids;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        ids.push_back(std::stoll(item));
    }
    return ids;
}

int main(int argc, char *argv[]) {

    // No buffering of printf
    setbuf(stdout, nullptr);

    // Retrieve run directory
    CommandLine cmd;
    std::string run_dir = "";
    cmd.Usage("Usage: ./waf --run=\"main_satnet --run_dir='<path/to/run/directory>'\"");
    cmd.AddValue("run_dir",  "Run directory", run_dir);
    cmd.Parse(argc, argv);
    if (run_dir.compare("") == 0) {
        printf("Usage: ./waf --run=\"main_satnet --run_dir='<path/to/run/directory>'\"");
        return 0;
    }

    // Load basic simulation environment
    Ptr<BasicSimulation> basicSimulation = CreateObject<BasicSimulation>(run_dir);

    // Setting socket type
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType", StringValue ("ns3::" + basicSimulation->GetConfigParamOrFail("tcp_socket_type")));

    // Optimize TCP
    TcpOptimizer::OptimizeBasic(basicSimulation);

    // Read topology, and install routing arbiters
    Ptr<TopologySatelliteNetwork> topology = CreateObject<TopologySatelliteNetwork>(basicSimulation, Ipv4ArbiterRoutingHelper());

    // Grayhole attack injection (optional, driven by config_ns3.properties)
    std::string grayhole_satellites = basicSimulation->GetConfigParamOrDefault("grayhole_satellites", "");
    if (!grayhole_satellites.empty()) {
        double grayhole_drop_rate = std::stod(basicSimulation->GetConfigParamOrFail("grayhole_drop_rate"));
        std::string grayhole_mode = basicSimulation->GetConfigParamOrDefault("grayhole_mode", "CONSTANT");
        Time grayhole_start = NanoSeconds(parse_positive_int64(basicSimulation->GetConfigParamOrDefault("grayhole_start_ns", "0")));
        Time grayhole_on = NanoSeconds(parse_positive_int64(basicSimulation->GetConfigParamOrDefault("grayhole_on_ns", "1000000000")));
        Time grayhole_off = NanoSeconds(parse_positive_int64(basicSimulation->GetConfigParamOrDefault("grayhole_off_ns", "1000000000")));

        std::vector<int64_t> evil_ids = ParseSatelliteIds(grayhole_satellites);
        for (int64_t sat_id : evil_ids) {
            Ptr<Node> node = topology->GetSatelliteNodes().Get(sat_id);
            for (uint32_t i = 0; i < node->GetNDevices(); i++) {
                Ptr<NetDevice> dev = node->GetDevice(i);
                Ptr<GrayholeErrorModel> em = CreateObject<GrayholeErrorModel>();
                em->SetAttribute("DropRate", DoubleValue(grayhole_drop_rate));
                em->SetAttribute("StartTime", TimeValue(grayhole_start));
                em->SetMode(GrayholeErrorModel::ParseMode(grayhole_mode));
                em->SetOnDuration(grayhole_on);
                em->SetOffDuration(grayhole_off);

                Ptr<PointToPointLaserNetDevice> laser = DynamicCast<PointToPointLaserNetDevice>(dev);
                if (laser != 0) {
                    laser->SetReceiveErrorModel(em);
                } else {
                    Ptr<GSLNetDevice> gsl = DynamicCast<GSLNetDevice>(dev);
                    if (gsl != 0) {
                        gsl->SetReceiveErrorModel(em);
                    }
                }
            }
            std::cout << "  > Injected grayhole on satellite " << sat_id
                      << " (drop_rate=" << grayhole_drop_rate
                      << ", mode=" << grayhole_mode << ")" << std::endl;
        }
    }

    ArbiterSingleForwardHelper arbiterHelper(basicSimulation, topology->GetNodes());
    GslIfBandwidthHelper gslIfBandwidthHelper(basicSimulation, topology->GetNodes());

    // Schedule flows
    TcpFlowScheduler tcpFlowScheduler(basicSimulation, topology); // Requires enable_tcp_flow_scheduler=true

    // Schedule UDP bursts
    UdpBurstScheduler udpBurstScheduler(basicSimulation, topology); // Requires enable_udp_burst_scheduler=true

    // Schedule pings
    PingmeshScheduler pingmeshScheduler(basicSimulation, topology); // Requires enable_pingmesh_scheduler=true

    // Run simulation
    basicSimulation->Run();

    // Write flow results
    tcpFlowScheduler.WriteResults();

    // Write UDP burst results
    udpBurstScheduler.WriteResults();

    // Write pingmesh results
    pingmeshScheduler.WriteResults();

    // Collect utilization statistics
    topology->CollectUtilizationStatistics();

    // Finalize the simulation
    basicSimulation->Finalize();

    return 0;

}
