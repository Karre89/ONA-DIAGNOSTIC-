'use client'

import { motion } from 'framer-motion'
import {
  Activity,
  ArrowRight,
  CheckCircle,
  Globe,
  Heart,
  Lock,
  Mail,
  MapPin,
  Shield,
  Smartphone,
  Sparkles,
  WifiOff,
  Zap,
} from 'lucide-react'
import Link from 'next/link'

const features = [
  {
    icon: Zap,
    title: 'Multi-Condition Detection',
    description: 'AI trained on 600K+ chest X-rays. Detects TB, Pneumonia, Cardiomegaly, and 10+ other conditions.',
  },
  {
    icon: WifiOff,
    title: 'Works Offline',
    description: 'Process scans locally without internet. Results sync automatically when connected.',
  },
  {
    icon: Lock,
    title: 'Privacy First',
    description: 'Patient data stays local. Only de-identified results sync to the cloud.',
  },
  {
    icon: Globe,
    title: 'Multi-Language',
    description: 'Support for English, Swahili, and Somali. More languages coming soon.',
  },
  {
    icon: Smartphone,
    title: 'Easy to Use',
    description: 'Simple interface designed for busy healthcare workers. Minimal training required.',
  },
  {
    icon: Shield,
    title: 'Clinical Grade',
    description: 'Built with healthcare compliance in mind. Full audit trails and secure data handling.',
  },
]

const stats = [
  { value: '600K+', label: 'Training Images' },
  { value: '14', label: 'Conditions Detected' },
  { value: '<2s', label: 'Processing Time' },
  { value: '3', label: 'Languages' },
]

const conditions = [
  { condition: 'Tuberculosis', icon: '🫁' },
  { condition: 'Pneumonia', icon: '🦠' },
  { condition: 'Cardiomegaly', icon: '❤️' },
  { condition: 'Pleural Effusion', icon: '💧' },
  { condition: 'Atelectasis', icon: '🔬' },
  { condition: 'Consolidation', icon: '📊' },
]

const fadeUp = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.7, ease: [0.4, 0, 0.2, 1] },
}

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.08,
    },
  },
}

export default function AboutPage() {
  return (
    <div className="min-h-screen relative">
      {/* Background */}
      <div className="fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-b from-[#060d18] via-[#0a1628] to-[#0d1b30]" />
        <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-[#00CED1]/[0.03] rounded-full blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-[#20B2AA]/[0.03] rounded-full blur-[120px]" />
      </div>

      {/* Navbar */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass-strong">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link href="/" className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#00CED1] to-[#20B2AA] flex items-center justify-center shadow-lg shadow-[#00CED1]/20">
                <Activity className="w-4 h-4 text-white" />
              </div>
              <span className="text-lg font-semibold text-white tracking-tight">ONA Health</span>
            </Link>
            <div className="hidden md:flex items-center gap-8">
              <a href="#features" className="text-sm text-gray-400 hover:text-white transition-colors duration-300">Features</a>
              <a href="#about" className="text-sm text-gray-400 hover:text-white transition-colors duration-300">About</a>
              <a href="#contact" className="text-sm text-gray-400 hover:text-white transition-colors duration-300">Contact</a>
            </div>
            <div className="flex items-center gap-4">
              <Link href="/" className="text-sm text-gray-400 hover:text-white transition-colors duration-300">
                Sign In
              </Link>
              <Link href="/" className="btn-primary text-sm py-2 px-5">
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-36 pb-24 px-6 relative overflow-hidden">
        <div className="max-w-7xl mx-auto text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1] }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-8">
              <Heart className="w-3.5 h-3.5 text-red-400" />
              <span className="text-xs text-gray-400 tracking-wide uppercase">Saving lives through early detection</span>
            </div>
            <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold text-white mb-6 tracking-tight leading-[0.95]">
              AI-Powered<br />
              <span className="text-shimmer">Medical Imaging</span>
            </h1>
            <p className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed">
              Edge-first diagnostics for Africa. Fast, accurate, offline-capable screening
              for TB, Pneumonia, Cardiac conditions and more.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Link href="/" className="btn-primary flex items-center gap-2 text-sm">
                Get Started <ArrowRight className="w-4 h-4" />
              </Link>
              <a href="#features" className="glass px-6 py-3 rounded-xl text-sm text-gray-300 hover:text-white hover:border-white/10 transition-all duration-300">
                Learn More
              </a>
            </div>
          </motion.div>

          {/* Stats */}
          <motion.div
            variants={staggerContainer}
            initial="initial"
            animate="animate"
            className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-24"
          >
            {stats.map((stat) => (
              <motion.div
                key={stat.label}
                variants={fadeUp}
                className="glass-card rounded-2xl p-6 glow-pulse"
              >
                <p className="text-3xl md:text-4xl font-bold text-gradient tracking-tight">{stat.value}</p>
                <p className="text-gray-500 text-sm mt-1.5 tracking-wide">{stat.label}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      <div className="divider-glow max-w-4xl mx-auto" />

      {/* Features Section */}
      <section id="features" className="py-24 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="text-center mb-16"
          >
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full glass mb-6">
              <Sparkles className="w-3.5 h-3.5 text-[#00CED1]" />
              <span className="text-xs text-gray-400 tracking-wide uppercase">Features</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4 tracking-tight">
              Built for Healthcare Workers
            </h2>
            <p className="text-gray-400 max-w-xl mx-auto leading-relaxed">
              Every feature designed with clinical workflows in mind. Simple, fast, reliable.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.08, duration: 0.6 }}
                className="glass-card rounded-2xl p-6 group cursor-default"
              >
                <div className="icon-container w-11 h-11 rounded-xl flex items-center justify-center mb-4">
                  <feature.icon className="w-5 h-5 text-[#00CED1]" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2 tracking-tight">{feature.title}</h3>
                <p className="text-gray-400 text-sm leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <div className="divider-glow max-w-4xl mx-auto" />

      {/* About / Mission Section */}
      <section id="about" className="py-24 px-6 relative">
        <div className="max-w-7xl mx-auto relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.7 }}
            >
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full glass mb-6">
                <Heart className="w-3.5 h-3.5 text-red-400" />
                <span className="text-xs text-gray-400 tracking-wide uppercase">Our Mission</span>
              </div>
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-6 tracking-tight">
                Diagnostics for <span className="text-gradient">Everyone</span>
              </h2>
              <p className="text-gray-400 mb-5 leading-relaxed">
                Millions die each year from treatable conditions — TB, pneumonia, heart disease —
                simply because their local clinic lacks a specialist to read a scan.
                Early detection saves lives, but access to expertise remains limited.
              </p>
              <p className="text-gray-400 mb-8 leading-relaxed">
                ONA Health brings AI-powered diagnostics to the point of care. Our edge-first approach
                means clinics can screen for 14+ conditions even without reliable internet, ensuring
                no one is left behind.
              </p>
              <div className="space-y-3.5">
                {[
                  'Screen for TB, pneumonia, cardiac conditions and more',
                  'Results in under 2 minutes, works offline',
                  'Multi-language support (English, Swahili, Somali)',
                  'Seamless integration with any X-ray equipment',
                ].map((item) => (
                  <div key={item} className="flex items-center gap-3">
                    <div className="w-5 h-5 rounded-full bg-[#00CED1]/10 flex items-center justify-center flex-shrink-0">
                      <CheckCircle className="w-3.5 h-3.5 text-[#00CED1]" />
                    </div>
                    <span className="text-gray-300 text-sm">{item}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.7, delay: 0.1 }}
              className="glass-card rounded-3xl p-8"
            >
              <h3 className="text-xl font-semibold text-white mb-6 tracking-tight">What We Detect</h3>
              <div className="grid grid-cols-2 gap-3">
                {conditions.map((item) => (
                  <div key={item.condition} className="flex items-center gap-3 p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.04] hover:border-white/[0.08] transition-colors duration-300">
                    <span className="text-lg">{item.icon}</span>
                    <span className="text-gray-300 text-sm">{item.condition}</span>
                  </div>
                ))}
              </div>
              <p className="text-gray-600 text-xs mt-5 text-center tracking-wide">
                + 8 more conditions powered by TorchXRayVision
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      <div className="divider-glow max-w-4xl mx-auto" />

      {/* Contact Section */}
      <section id="contact" className="py-24 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="glass-card rounded-3xl p-8 md:p-12"
          >
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full glass mb-6">
                  <Mail className="w-3.5 h-3.5 text-[#00CED1]" />
                  <span className="text-xs text-gray-400 tracking-wide uppercase">Contact</span>
                </div>
                <h2 className="text-3xl md:text-4xl font-bold text-white mb-4 tracking-tight">Get in Touch</h2>
                <p className="text-gray-400 mb-8 leading-relaxed text-sm">
                  Interested in bringing ONA Health to your facility? Let&apos;s talk about how we can help.
                </p>
                <div className="space-y-5">
                  <div className="flex items-center gap-4">
                    <div className="icon-container w-10 h-10 rounded-xl flex items-center justify-center">
                      <Mail className="w-4 h-4 text-[#00CED1]" />
                    </div>
                    <div>
                      <p className="text-gray-500 text-xs tracking-wide uppercase">Email</p>
                      <p className="text-white text-sm">kayse@onahealth.africa</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="icon-container w-10 h-10 rounded-xl flex items-center justify-center">
                      <Globe className="w-4 h-4 text-[#00CED1]" />
                    </div>
                    <div>
                      <p className="text-gray-500 text-xs tracking-wide uppercase">Website</p>
                      <p className="text-white text-sm">onahealth.africa</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="icon-container w-10 h-10 rounded-xl flex items-center justify-center">
                      <MapPin className="w-4 h-4 text-[#00CED1]" />
                    </div>
                    <div>
                      <p className="text-gray-500 text-xs tracking-wide uppercase">Location</p>
                      <p className="text-white text-sm">Nairobi, Kenya</p>
                    </div>
                  </div>
                </div>
              </div>

              <form className="space-y-4" onSubmit={(e) => e.preventDefault()}>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <input type="text" placeholder="Your Name" className="input-glass" />
                  <input type="email" placeholder="Email Address" className="input-glass" />
                </div>
                <input type="text" placeholder="Organization" className="input-glass" />
                <textarea placeholder="Your Message" rows={4} className="input-glass resize-none" />
                <button type="submit" className="btn-primary w-full text-sm">
                  Send Message
                </button>
              </form>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-6 border-t border-white/[0.04]">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-[#00CED1] to-[#20B2AA] flex items-center justify-center">
              <Activity className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="text-gray-500 text-sm">&copy; 2026 ONA Health. All rights reserved.</span>
          </div>
          <p className="text-[#00CED1]/60 text-sm font-medium tracking-wide">See Clearly. Act Quickly.</p>
        </div>
      </footer>
    </div>
  )
}
