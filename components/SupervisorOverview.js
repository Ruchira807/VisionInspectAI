import { useEffect, useState } from 'react';
import { downloadCSV } from '../lib/csv';
import { getSupervisorHistoryPDF } from '../lib/api';

function SkeletonBlock({ className = '' }) {
  return <div className={`bg-gridline/40 animate-pulse ${className}`} />;
}

export default function SupervisorOverview() {
  const [isLoading, setIsLoading] = useState(true);
  const [queue, setQueue] = useState([]);
  const [trend, setTrend] = useState([]);
  const [defects, setDefects] = useState([]);
  const [production, setProduction] = useState(null);
  const [risk, setRisk] = useState({});

  useEffect(() => {
    async function loadData() {
      setIsLoading(true);

      try {
        const BASE =
          process.env.NEXT_PUBLIC_API_URL ||
          'http://127.0.0.1:8000';

        const token = localStorage.getItem('vi_token');

        if (!token) {
          throw new Error('Authentication required');
        }

        const authHeaders = {
          Authorization: `Bearer ${token}`,
        };

        const [
          productionRes,
          trendRes,
          riskRes,
          historyRes,
          workflowRes,
        ] = await Promise.all([
          fetch(`${BASE}/reports/production`, {
            headers: authHeaders,
          }),

          fetch(`${BASE}/analytics/trends`, {
            headers: authHeaders,
          }),

          fetch(`${BASE}/analytics/risk-assessment`, {
            headers: authHeaders,
          }),

          fetch(`${BASE}/admin/history`, {
            headers: authHeaders,
          }),

          fetch(`${BASE}/supervisor/workflows`, {
            headers: authHeaders,
          }),
        ]);

        /*
         * IMPORTANT:
         *
         * /history is the source of truth for the supervisor.
         *
         * The backend correctly returns ALL inspections for the
         * supervisor/admin account.
         *
         * Some analytics endpoints may be user-scoped, so we do
         * NOT rely on /reports/production for the supervisor's
         * inspection count/pass rate/severity distribution.
         */

        if (!historyRes.ok) {
          throw new Error('Failed to fetch inspection history');
        }

        const historyData = await historyRes.json();

        const workflowData = workflowRes.ok
          ? await workflowRes.json()
          : [];

        /*
         * ----------------------------------------------------------
         * SUPERVISOR-WIDE PRODUCTION STATISTICS
         * ----------------------------------------------------------
         */

        const totalInspections = historyData.length;

        const passedInspections = historyData.filter(
          (item) => item.result === 'PASS'
        ).length;

        const passRate =
          totalInspections > 0
            ? Number(
                ((passedInspections / totalInspections) * 100).toFixed(1)
              )
            : 0;

        /*
         * Severity distribution
         */
        const severityDistribution = {};

        historyData.forEach((item) => {
          const severity = item.severity_level || 'Unknown';

          severityDistribution[severity] =
            (severityDistribution[severity] || 0) + 1;
        });

        /*
         * Category statistics
         */
        const categoryStats = {};

        historyData.forEach((item) => {
          const category = item.category || 'Unknown';

          if (!categoryStats[category]) {
            categoryStats[category] = {
              total: 0,
              defects: 0,
            };
          }

          categoryStats[category].total += 1;

          if (item.result === 'REJECT') {
            categoryStats[category].defects += 1;
          }
        });

        /*
         * Build production object locally from ALL supervisor history.
         */
        const supervisorProduction = {
          total_units_inspected: totalInspections,
          yield_pass_rate_pct: passRate,
          severity_distribution: severityDistribution,
        };

        setProduction(supervisorProduction);

        /*
         * ----------------------------------------------------------
         * RISK / PRODUCTION LINE HEALTH
         * ----------------------------------------------------------
         */

        const supervisorRiskLevels = {};

        Object.entries(categoryStats).forEach(
          ([category, stats]) => {
            const defectRate =
              stats.total > 0
                ? (stats.defects / stats.total) * 100
                : 0;

            let riskLevel = 'LOW RISK';

            if (defectRate >= 30) {
              riskLevel = 'HIGH RISK';
            } else if (defectRate >= 10) {
              riskLevel = 'MEDIUM RISK';
            }

            supervisorRiskLevels[category] = {
              risk_level: riskLevel,
              defect_rate_pct: Number(defectRate.toFixed(1)),
              total_inspections: stats.total,
              defects: stats.defects,
            };
          }
        );

        setRisk({
          category_risk_levels: supervisorRiskLevels,
        });

        /*
         * ----------------------------------------------------------
         * PASS RATE TREND
         * ----------------------------------------------------------
         *
         * historyData is returned newest -> oldest.
         * Reverse it so the trend runs chronologically.
         */

        let passed = 0;

        const trendDataPoints = [...historyData]
          .reverse()
          .map((item, index) => {
            if (item.result === 'PASS') {
              passed += 1;
            }

            const total = index + 1;

            return {
              passRate: Number(
                ((passed / total) * 100).toFixed(1)
              ),
            };
          });

        setTrend(trendDataPoints.slice(-7));

        /*
         * ----------------------------------------------------------
         * DEFECT DISTRIBUTION
         * ----------------------------------------------------------
         */

        setDefects(
          Object.entries(severityDistribution)
            .filter(([name]) => name !== 'Unknown')
            .map(([name, count]) => ({
              name,
              count,
            }))
        );

        /*
         * ----------------------------------------------------------
         * ESCALATION QUEUE
         * ----------------------------------------------------------
         */

        const handledInspectionIds = new Set(
          workflowData
            .filter(
              (workflow) =>
                workflow.status === 'REWORK_APPROVED' ||
                workflow.status === 'ESCALATED'
            )
            .map((workflow) => workflow.inspection_id)
        );

        const critical = historyData.filter(
          (item) =>
            (item.severity_level === 'Critical' ||
              item.severity_level === 'High') &&
            !handledInspectionIds.has(item.id)
        );

        setQueue(
          critical.map((item, index) => ({
            id: item.id ?? index,
            product: item.category,
            defect: item.defect,
            severity: item.severity_score,
            severityLevel: item.severity_level,
            line: 'Production Line',
          }))
        );
      } catch (err) {
        console.error(
          'Failed to load supervisor overview data:',
          err
        );
      } finally {
        setIsLoading(false);
      }
    }

    loadData();
  }, []);

  const maxDefect =
    defects.length > 0
      ? Math.max(...defects.map((d) => d.count))
      : 1;

  async function handleDownloadPDF() {
  try {
    const blob = await getSupervisorHistoryPDF();

    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');

    link.href = url;
    link.download = 'all_inspection_history.pdf';
    link.click();

    window.URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Failed to download supervisor PDF:', error);
  }
  }

  async function resolve(id, action) {
    try {
      const BASE =
        process.env.NEXT_PUBLIC_API_URL ||
        'http://127.0.0.1:8000';

      const endpoint =
        action === 'approve'
          ? `${BASE}/supervisor/inspections/${id}/approve-rework`
          : `${BASE}/supervisor/inspections/${id}/escalate`;

      const token = localStorage.getItem('vi_token');

      if (!token) {
        throw new Error('Authentication required');
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        throw new Error(
          'Failed to update inspection workflow'
        );
      }

      setQueue((prev) =>
        prev.filter((item) => item.id !== id)
      );
    } catch (err) {
      console.error(
        'Failed to update supervisor action:',
        err
      );
    }
  }

  function handleExport() {
    downloadCSV('escalation_queue.csv', queue, [
      { key: 'product', label: 'Product' },
      { key: 'line', label: 'Line' },
      { key: 'defect', label: 'Defect' },
      {
        key: 'severityLevel',
        label: 'Severity Level',
      },
      {
        key: 'severity',
        label: 'Severity Score',
      },
    ]);
  }

  return (
    <div className="space-y-6">

      {/* ----------------------------------------------------------
          KPI CARDS
      ---------------------------------------------------------- */}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">

        <div className="bg-panel border border-gridline p-4">
          <p className="text-xs font-mono text-muted uppercase">
            Total Inspections
          </p>

          {isLoading ? (
            <SkeletonBlock className="h-7 w-14 mt-2" />
          ) : (
            <p className="font-display text-2xl text-ink mt-1">
              {production?.total_units_inspected ?? 0}
            </p>
          )}
        </div>

        <div className="bg-panel border border-gridline p-4">
          <p className="text-xs font-mono text-muted uppercase">
            Pass Rate
          </p>

          {isLoading ? (
            <SkeletonBlock className="h-7 w-14 mt-2" />
          ) : (
            <p className="font-display text-2xl text-ok mt-1">
              {production?.yield_pass_rate_pct ?? 0}%
            </p>
          )}
        </div>

        <div className="bg-panel border border-gridline p-4">
          <p className="text-xs font-mono text-muted uppercase">
            Critical Defects
          </p>

          {isLoading ? (
            <SkeletonBlock className="h-7 w-14 mt-2" />
          ) : (
            <p className="font-display text-2xl text-signal mt-1">
              {production?.severity_distribution?.Critical ?? 0}
            </p>
          )}
        </div>

        <div className="bg-panel border border-gridline p-4">
          <p className="text-xs font-mono text-muted uppercase">
            Categories Monitored
          </p>

          {isLoading ? (
            <SkeletonBlock className="h-7 w-14 mt-2" />
          ) : (
            <p className="font-display text-2xl text-ink mt-1">
              {Object.keys(
                risk.category_risk_levels || {}
              ).length}
            </p>
          )}
        </div>

      </div>

      {/* ----------------------------------------------------------
          PRODUCTION LINE HEALTH
      ---------------------------------------------------------- */}

      <div className="bg-panel border border-gridline p-4 sm:p-6">
        <h2 className="font-display text-lg text-ink mb-4">
          Production Line Health
        </h2>

        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <SkeletonBlock className="h-20 w-full" />
            <SkeletonBlock className="h-20 w-full" />
            <SkeletonBlock className="h-20 w-full" />
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

            {Object.entries(
              risk.category_risk_levels || {}
            ).map(([name, line]) => (

              <div
                key={name}
                className="bg-graphite border border-gridline p-4"
              >
                <div className="flex items-center justify-between mb-2">

                  <span className="text-sm text-ink">
                    {name}
                  </span>

                  <span
                    className={`text-xs font-mono px-2 py-1 border ${
                      line.risk_level === 'HIGH RISK'
                        ? 'border-signal text-signal'
                        : line.risk_level === 'MEDIUM RISK'
                        ? 'border-warn text-warn'
                        : 'border-ok text-ok'
                    }`}
                  >
                    {line.risk_level}
                  </span>

                </div>

                <div className="w-full h-2 bg-gridline">
                  <div
                    className={`h-full ${
                      line.risk_level === 'HIGH RISK'
                        ? 'bg-signal'
                        : line.risk_level === 'MEDIUM RISK'
                        ? 'bg-warn'
                        : 'bg-ok'
                    }`}
                    style={{
                      width: `${Math.max(
                        0,
                        Math.min(
                          100,
                          100 - line.defect_rate_pct
                        )
                      )}%`,
                    }}
                  />
                </div>

                <p className="text-xs font-mono text-muted mt-2">
                  {(
                    100 - line.defect_rate_pct
                  ).toFixed(1)}
                  % pass rate
                </p>
              </div>

            ))}

          </div>
        )}
      </div>

      {/* ----------------------------------------------------------
          TRENDS + DEFECT DISTRIBUTION
      ---------------------------------------------------------- */}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

        <div className="bg-panel border border-gridline p-4 sm:p-6">
          <h2 className="font-display text-lg text-ink mb-4">
            Recent Pass Rate Trend
          </h2>

          <div className="flex items-end gap-3 h-32">

            {isLoading ? (
              Array.from({ length: 7 }).map((_, i) => (
                <div
                  key={i}
                  className="flex-1 h-full flex flex-col items-center justify-end gap-2"
                >
                  <SkeletonBlock className="w-full flex-1" />

                  <span className="text-[10px] font-mono text-muted">
                    D{i + 1}
                  </span>
                </div>
              ))
            ) : trend.length === 0 ? (
              <div className="w-full text-center text-sm text-muted">
                No trend data available.
              </div>
            ) : (
              trend.map((val, i) => (
                <div
                  key={i}
                  className="flex-1 h-full flex flex-col items-center justify-end gap-2"
                >
                  <div
                    className="w-full bg-signal/40 transition-all"
                    style={{
                      height: `${Math.max(
                        val.passRate,
                        5
                      )}%`,
                    }}
                    title={`Pass rate: ${val.passRate}%`}
                  />

                  <span className="text-[10px] font-mono text-muted">
                    {val.passRate}%
                  </span>
                </div>
              ))
            )}

          </div>
        </div>

        <div className="bg-panel border border-gridline p-4 sm:p-6">
          <h2 className="font-display text-lg text-ink mb-4">
            Plant-Wide Defect Distribution
          </h2>

          <div className="space-y-3">

            {isLoading ? (
              <>
                <SkeletonBlock className="h-2 w-full" />
                <SkeletonBlock className="h-2 w-full" />
                <SkeletonBlock className="h-2 w-full" />
              </>
            ) : (
              defects.map((d) => (
                <div key={d.name}>

                  <div className="flex items-center justify-between text-xs font-mono text-muted mb-1">
                    <span>{d.name}</span>
                    <span>{d.count}</span>
                  </div>

                  <div className="w-full h-2 bg-graphite">
                    <div
                      className="h-full bg-signal"
                      style={{
                        width: `${
                          (d.count / maxDefect) * 100
                        }%`,
                      }}
                    />
                  </div>

                </div>
              ))
            )}

          </div>
        </div>

      </div>

      {/* ----------------------------------------------------------
          ESCALATION QUEUE
      ---------------------------------------------------------- */}

      <div className="bg-panel border border-gridline">

        <div className="flex flex-wrap items-center justify-between gap-3 px-4 sm:px-6 py-4 border-b border-gridline">

          <h2 className="font-display text-lg text-ink">
            Escalation Queue
          </h2>

          <div className="flex items-center gap-3">

            <span className="text-xs font-mono text-muted uppercase">
              {isLoading
                ? '...'
                : `${queue.length} pending`}
            </span>

            <button
              onClick={handleExport}
              disabled={!queue.length}
              className="text-xs font-mono border border-gridline px-3 py-1.5 text-muted hover:border-signal hover:text-ink transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Export CSV
            </button>

            <button
              onClick={handleDownloadPDF}
              className="text-xs font-mono border border-gridline px-3 py-1.5 text-muted hover:border-signal hover:text-ink transition-colors"
            >
              Download PDF
            </button>

          </div>
        </div>

        {isLoading ? (
          <div className="px-4 sm:px-6 py-4 space-y-3">
            <SkeletonBlock className="h-14 w-full" />
            <SkeletonBlock className="h-14 w-full" />
          </div>
        ) : queue.length === 0 ? (
          <div className="px-6 py-10 text-center text-sm text-muted font-body">
            No pending escalations — all clear for now.
          </div>
        ) : (
          <div>

            {queue.map((item) => (
              <div
                key={item.id}
                className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-4 sm:px-6 py-4 border-b border-gridline last:border-0"
              >

                <div>
                  <p className="text-ink text-sm font-body">
                    {item.product}
                    {' · '}
                    <span className="text-muted">
                      {item.line}
                    </span>
                  </p>

                  <p className="text-xs font-mono text-signal mt-1">
                    {item.defect}
                    {' — '}
                    {item.severityLevel}
                    {' · severity '}
                    {item.severity}
                  </p>
                </div>

                <div className="flex gap-2">

                  <button
                    onClick={() =>
                      resolve(item.id, 'approve')
                    }
                    className="flex-1 sm:flex-none text-xs font-mono border border-gridline px-3 py-1.5 text-muted hover:border-ok hover:text-ok transition-colors"
                  >
                    Approve Rework
                  </button>

                  <button
                    onClick={() =>
                      resolve(item.id, 'escalate')
                    }
                    className="flex-1 sm:flex-none text-xs font-mono border border-signal px-3 py-1.5 text-signal hover:bg-signal/10 transition-colors"
                  >
                    Escalate
                  </button>

                </div>

              </div>
            ))}

          </div>
        )}

      </div>

    </div>
  );
}