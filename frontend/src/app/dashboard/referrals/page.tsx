'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  CheckCircle,
  Clock,
  FileText,
  Search,
  User,
  MessageSquare,
} from 'lucide-react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { api, ReferralInfo, ReferralStats } from '@/lib/api'

const STATUS_CONFIG: Record<string, { color: string; label: string }> = {
  pending: { color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30', label: 'Pending' },
  arrived: { color: 'bg-blue-500/20 text-blue-400 border-blue-500/30', label: 'Arrived' },
  scanned: { color: 'bg-purple-500/20 text-purple-400 border-purple-500/30', label: 'Scanned' },
  completed: { color: 'bg-green-500/20 text-green-400 border-green-500/30', label: 'Completed' },
  no_show: { color: 'bg-red-500/20 text-red-400 border-red-500/30', label: 'No Show' },
}

const URGENCY_CONFIG: Record<string, string> = {
  LOW: 'bg-green-500/20 text-green-400',
  MEDIUM: 'bg-yellow-500/20 text-yellow-400',
  HIGH: 'bg-orange-500/20 text-orange-400',
  CRITICAL: 'bg-red-500/20 text-red-400',
}

export default function ReferralsPage() {
  const router = useRouter()
  const [searchCode, setSearchCode] = useState('')
  const [lookupResult, setLookupResult] = useState<ReferralInfo | null>(null)
  const [referrals, setReferrals] = useState<ReferralInfo[]>([])
  const [stats, setStats] = useState<ReferralStats | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSearching, setIsSearching] = useState(false)
  const [error, setError] = useState('')
  const [searchError, setSearchError] = useState('')

  useEffect(() => {
    if (!api.isAuthenticated()) {
      router.push('/')
      return
    }
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      setIsLoading(true)
      const [referralList, referralStats] = await Promise.all([
        api.listReferrals().catch(() => []),
        api.getReferralStats().catch(() => null),
      ])
      setReferrals(referralList)
      setStats(referralStats)
      setError('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load referrals')
    } finally {
      setIsLoading(false)
    }
  }

  const handleLookup = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!searchCode.trim()) return

    try {
      setIsSearching(true)
      setSearchError('')
      setLookupResult(null)
      const result = await api.lookupReferral(searchCode.trim())
      setLookupResult(result)
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : 'Referral not found')
    } finally {
      setIsSearching(false)
    }
  }

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return 'N/A'
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    })
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <nav className="glass-strong border-b border-white/5 sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link href="/dashboard" className="p-2 rounded-lg hover:bg-white/5 transition-colors">
                <ArrowLeft className="w-5 h-5 text-gray-400" />
              </Link>
              <div>
                <h1 className="text-lg font-bold text-white">Referral Lookup</h1>
                <p className="text-xs text-gray-400">SYNARA Referral Pipeline</p>
              </div>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Search */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card rounded-2xl p-6 mb-8"
        >
          <h3 className="text-lg font-semibold text-white mb-4">Look Up Referral Code</h3>
          <form onSubmit={handleLookup} className="flex gap-3">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
              <input
                type="text"
                value={searchCode}
                onChange={(e) => setSearchCode(e.target.value.toUpperCase())}
                placeholder="Enter referral code (e.g. ONA-A1B2C3)"
                className="input-glass pl-10 uppercase tracking-wider"
              />
            </div>
            <button
              type="submit"
              disabled={isSearching || !searchCode.trim()}
              className="btn-primary flex items-center gap-2 whitespace-nowrap"
            >
              {isSearching ? (
                <Activity className="w-4 h-4 animate-spin" />
              ) : (
                <Search className="w-4 h-4" />
              )}
              Look Up
            </button>
          </form>

          {searchError && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-4 p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm flex items-center gap-2"
            >
              <AlertTriangle className="w-4 h-4" />
              {searchError}
            </motion.div>
          )}

          {/* Lookup Result */}
          {lookupResult && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-6 p-5 rounded-xl bg-white/5 border border-white/10"
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h4 className="text-white font-bold text-lg tracking-wider">{lookupResult.referral_code}</h4>
                  <p className="text-sm text-gray-400 capitalize">{lookupResult.suspected_condition}</p>
                </div>
                <div className="flex gap-2">
                  <span className={`px-3 py-1 rounded-lg text-xs font-medium ${URGENCY_CONFIG[lookupResult.urgency] || 'bg-gray-500/20 text-gray-400'}`}>
                    {lookupResult.urgency}
                  </span>
                  <span className={`px-3 py-1 rounded-lg text-xs font-medium border ${STATUS_CONFIG[lookupResult.status]?.color || 'bg-gray-500/20 text-gray-400'}`}>
                    {STATUS_CONFIG[lookupResult.status]?.label || lookupResult.status}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm">
                {lookupResult.symptoms && lookupResult.symptoms.length > 0 && (
                  <div>
                    <p className="text-gray-500 mb-1">Symptoms</p>
                    <div className="flex flex-wrap gap-1">
                      {lookupResult.symptoms.map((s, i) => (
                        <span key={i} className="px-2 py-0.5 bg-white/5 rounded text-gray-300 text-xs">{s}</span>
                      ))}
                    </div>
                  </div>
                )}
                {lookupResult.patient_language && (
                  <div>
                    <p className="text-gray-500 mb-1">Language</p>
                    <p className="text-white">{lookupResult.patient_language}</p>
                  </div>
                )}
                {lookupResult.triage_confidence !== null && (
                  <div>
                    <p className="text-gray-500 mb-1">Triage Confidence</p>
                    <p className="text-white">{Math.round(lookupResult.triage_confidence * 100)}%</p>
                  </div>
                )}
                <div>
                  <p className="text-gray-500 mb-1">Referred</p>
                  <p className="text-white">{formatDate(lookupResult.referred_at || lookupResult.created_at)}</p>
                </div>
              </div>

              {lookupResult.scan_id && (
                <div className="mt-4 pt-4 border-t border-white/10 flex items-center justify-between">
                  <span className="text-sm text-gray-400">Scan linked to this referral</span>
                  <Link
                    href={`/dashboard/results/${lookupResult.scan_id}`}
                    className="btn-primary text-sm py-2 px-4 flex items-center gap-2"
                  >
                    View Scan <ArrowRight className="w-4 h-4" />
                  </Link>
                </div>
              )}
            </motion.div>
          )}
        </motion.div>

        {/* Stats */}
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {[
              { label: 'Total Referrals', value: stats.total, icon: FileText },
              { label: 'Pending', value: stats.pending, icon: Clock },
              { label: 'Completed', value: stats.completed, icon: CheckCircle },
              { label: 'Conversion', value: `${stats.conversion_rate}%`, icon: ArrowRight },
            ].map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 + index * 0.1 }}
                className="glass-card rounded-2xl p-5"
              >
                <stat.icon className="w-5 h-5 text-ona-primary mb-2" />
                <p className="text-2xl font-bold text-white">{stat.value}</p>
                <p className="text-sm text-gray-400">{stat.label}</p>
              </motion.div>
            ))}
          </div>
        )}

        {/* Referral List */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-card rounded-2xl p-6"
        >
          <h3 className="text-lg font-semibold text-white mb-6">Recent Referrals</h3>

          {isLoading ? (
            <div className="space-y-3">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="p-4 rounded-xl bg-white/5 animate-pulse h-16" />
              ))}
            </div>
          ) : referrals.length === 0 ? (
            <div className="text-center py-12 text-gray-400">
              <MessageSquare className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No referrals yet</p>
              <p className="text-sm mt-1">Referrals from SYNARA will appear here</p>
            </div>
          ) : (
            <div className="space-y-3">
              {referrals.map((ref, index) => (
                <motion.div
                  key={ref.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 + index * 0.05 }}
                  className="flex items-center justify-between p-4 rounded-xl bg-white/5 hover:bg-white/10 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <div className="text-white font-mono font-bold tracking-wider">{ref.referral_code}</div>
                    <div>
                      <p className="text-sm text-gray-300 capitalize">{ref.suspected_condition}</p>
                      <p className="text-xs text-gray-500">{formatDate(ref.created_at)}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`px-2 py-1 rounded-lg text-xs font-medium ${URGENCY_CONFIG[ref.urgency] || 'bg-gray-500/20 text-gray-400'}`}>
                      {ref.urgency}
                    </span>
                    <span className={`px-3 py-1 rounded-lg text-xs font-medium border ${STATUS_CONFIG[ref.status]?.color || 'bg-gray-500/20 text-gray-400'}`}>
                      {STATUS_CONFIG[ref.status]?.label || ref.status}
                    </span>
                    {ref.scan_id && (
                      <Link
                        href={`/dashboard/results/${ref.scan_id}`}
                        className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                      >
                        <ArrowRight className="w-4 h-4 text-ona-primary" />
                      </Link>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </motion.div>
      </main>
    </div>
  )
}
