'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Clock,
  Download,
  FileText,
  Cpu,
  Shield,
  Zap,
} from 'lucide-react'
import Link from 'next/link'
import { useParams, useRouter } from 'next/navigation'
import { api, ScanResult } from '@/lib/api'

const CONDITION_LABELS: Record<string, string> = {
  Atelectasis: 'Atelectasis',
  Cardiomegaly: 'Cardiomegaly',
  Consolidation: 'Consolidation',
  Edema: 'Edema',
  Effusion: 'Pleural Effusion',
  Emphysema: 'Emphysema',
  Fibrosis: 'Fibrosis',
  Hernia: 'Hernia',
  Infiltration: 'Infiltration',
  Mass: 'Mass',
  Nodule: 'Nodule',
  Pleural_Thickening: 'Pleural Thickening',
  Pneumonia: 'Pneumonia',
  Pneumothorax: 'Pneumothorax',
  tb: 'Tuberculosis',
  Tuberculosis: 'Tuberculosis',
  pneumonia: 'Pneumonia',
  cardiomegaly: 'Cardiomegaly',
}

const getRiskColor = (risk: string) => {
  switch (risk) {
    case 'HIGH': return 'bg-red-500/20 text-red-400 border-red-500/30'
    case 'MEDIUM': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
    case 'LOW': return 'bg-green-500/20 text-green-400 border-green-500/30'
    default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
  }
}

const getScoreColor = (score: number) => {
  if (score >= 0.6) return 'text-red-400'
  if (score >= 0.3) return 'text-yellow-400'
  return 'text-green-400'
}

const getBarColor = (score: number) => {
  if (score >= 0.6) return 'bg-red-500'
  if (score >= 0.3) return 'bg-yellow-500'
  return 'bg-green-500'
}

export default function ScanDetailPage() {
  const params = useParams()
  const router = useRouter()
  const [scan, setScan] = useState<ScanResult | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState('')
  const [downloading, setDownloading] = useState(false)

  const scanId = params.id as string

  useEffect(() => {
    if (!api.isAuthenticated()) {
      router.push('/')
      return
    }
    fetchScan()
  }, [scanId])

  const fetchScan = async () => {
    try {
      setIsLoading(true)
      const data = await api.getResult(scanId)
      setScan(data)
      setError('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load scan result')
    } finally {
      setIsLoading(false)
    }
  }

  const handleDownloadPdf = async () => {
    try {
      setDownloading(true)
      await api.downloadReport(scanId)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to download report')
    } finally {
      setDownloading(false)
    }
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 rounded-xl bg-ona-primary/20 flex items-center justify-center mx-auto mb-4 animate-pulse">
            <Activity className="w-6 h-6 text-ona-primary" />
          </div>
          <p className="text-gray-400">Loading scan result...</p>
        </div>
      </div>
    )
  }

  if (error && !scan) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="w-12 h-12 text-red-400 mx-auto mb-4" />
          <p className="text-red-400 mb-4">{error}</p>
          <Link href="/dashboard" className="text-ona-primary hover:underline">
            Back to Dashboard
          </Link>
        </div>
      </div>
    )
  }

  if (!scan) return null

  const sortedScores = Object.entries(scan.scores).sort(([, a], [, b]) => b - a)
  const flaggedConditions = sortedScores.filter(([, score]) => score >= 0.3)

  return (
    <div className="min-h-screen">
      {/* Header */}
      <nav className="glass-strong border-b border-white/5 sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link
                href="/dashboard"
                className="p-2 rounded-lg hover:bg-white/5 transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-gray-400" />
              </Link>
              <div>
                <h1 className="text-lg font-bold text-white">Scan Result</h1>
                <p className="text-xs text-gray-400">Study {scan.study_id.slice(0, 16)}</p>
              </div>
            </div>
            <button
              onClick={handleDownloadPdf}
              disabled={downloading}
              className="btn-primary flex items-center gap-2 text-sm"
            >
              <Download className={`w-4 h-4 ${downloading ? 'animate-bounce' : ''}`} />
              {downloading ? 'Generating...' : 'Download PDF'}
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 rounded-xl bg-yellow-500/10 border border-yellow-500/30 text-yellow-400 flex items-center gap-3"
          >
            <AlertTriangle className="w-5 h-5" />
            <span>{error}</span>
          </motion.div>
        )}

        {/* Top Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-card rounded-2xl p-5"
          >
            <div className="flex items-center gap-3 mb-2">
              <Shield className="w-5 h-5 text-ona-primary" />
              <span className="text-sm text-gray-400">Risk Level</span>
            </div>
            <div className={`inline-block px-3 py-1 rounded-lg text-sm font-bold border ${getRiskColor(scan.risk_bucket)}`}>
              {scan.risk_bucket}
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-card rounded-2xl p-5"
          >
            <div className="flex items-center gap-3 mb-2">
              <Clock className="w-5 h-5 text-ona-primary" />
              <span className="text-sm text-gray-400">Scan Date</span>
            </div>
            <p className="text-white font-medium text-sm">{formatDate(scan.created_at)}</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="glass-card rounded-2xl p-5"
          >
            <div className="flex items-center gap-3 mb-2">
              <Cpu className="w-5 h-5 text-ona-primary" />
              <span className="text-sm text-gray-400">Model</span>
            </div>
            <p className="text-white font-medium text-sm">{scan.model_version}</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="glass-card rounded-2xl p-5"
          >
            <div className="flex items-center gap-3 mb-2">
              <Zap className="w-5 h-5 text-ona-primary" />
              <span className="text-sm text-gray-400">Inference</span>
            </div>
            <p className="text-white font-medium text-sm">
              {scan.inference_time_ms ? `${scan.inference_time_ms}ms` : 'N/A'}
            </p>
          </motion.div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* All Condition Scores */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="lg:col-span-2 glass-card rounded-2xl p-6"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-white">AI Screening Results</h3>
              <span className="text-xs text-gray-500">{sortedScores.length} conditions analyzed</span>
            </div>

            <div className="space-y-3">
              {sortedScores.map(([condition, score], index) => (
                <motion.div
                  key={condition}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 + index * 0.03 }}
                  className="flex items-center gap-4"
                >
                  <div className="w-40 text-sm text-gray-300 truncate">
                    {CONDITION_LABELS[condition] || condition}
                  </div>
                  <div className="flex-1 h-6 bg-white/5 rounded-full overflow-hidden relative">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.max(score * 100, 1)}%` }}
                      transition={{ delay: 0.6 + index * 0.03, duration: 0.5 }}
                      className={`h-full rounded-full ${getBarColor(score)}`}
                      style={{ opacity: 0.7 }}
                    />
                  </div>
                  <div className={`w-14 text-right font-bold text-sm ${getScoreColor(score)}`}>
                    {Math.round(score * 100)}%
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Summary Panel */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="space-y-4"
          >
            {/* Flagged Conditions */}
            <div className="glass-card rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Flagged Conditions</h3>
              {flaggedConditions.length === 0 ? (
                <div className="text-center py-4">
                  <p className="text-green-400 font-medium">No conditions flagged</p>
                  <p className="text-xs text-gray-500 mt-1">All scores below threshold</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {flaggedConditions.map(([condition, score]) => (
                    <div key={condition} className="flex items-center justify-between p-3 rounded-xl bg-white/5">
                      <span className="text-sm text-white">
                        {CONDITION_LABELS[condition] || condition}
                      </span>
                      <span className={`text-sm font-bold ${getScoreColor(score)}`}>
                        {Math.round(score * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Explanation */}
            {scan.explanation && (
              <div className="glass-card rounded-2xl p-6">
                <h3 className="text-lg font-semibold text-white mb-3">AI Notes</h3>
                <p className="text-sm text-gray-300 leading-relaxed">{scan.explanation}</p>
              </div>
            )}

            {/* Scan Details */}
            <div className="glass-card rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Scan Details</h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Study ID</span>
                  <span className="text-white font-mono text-xs">{scan.study_id}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Modality</span>
                  <span className="text-white">{scan.modality}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Scan ID</span>
                  <span className="text-white font-mono text-xs">{scan.id.slice(0, 12)}...</span>
                </div>
              </div>
            </div>

            {/* Download */}
            <button
              onClick={handleDownloadPdf}
              disabled={downloading}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              <FileText className="w-5 h-5" />
              {downloading ? 'Generating PDF...' : 'Download Full Report'}
            </button>

            {/* Disclaimer */}
            <div className="p-4 rounded-xl bg-yellow-500/5 border border-yellow-500/10">
              <p className="text-xs text-yellow-400/70 leading-relaxed">
                This is an AI-assisted screening tool, not a clinical diagnosis.
                Results should be reviewed by a qualified healthcare professional.
              </p>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  )
}
