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
    description: 'Built with healthcare compliance in mind. Audit trails and secure data handling.',
  },
]

const stats = [
  { value: '600K+', label: 'Training Images' },
  { value: '14', label: 'Conditions Detected' },
  { value: '<2s', label: 'Processing Time' },
  { value: '3', label: 'Languages' },
]

export default function AboutPage() {
  return (
    <div className="min-h-screen">
      {/* Navbar */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass-strong border-b border-white/5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link href="/" className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-ona-primary to-ona-secondary flex items-center justify-center">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-white">ONA Health</span>
            </Link>
            <div className="hidden md:flex items-center gap-8">
              <a href="#features" className="text-gray-300 hover:text-white transition-colors">Features</a>
              <a href="#about" className="text-gray-300 hover:text-white transition-colors">About</a>
              <a href="#contact" className="text-gray-300 hover:text-white transition-colors">Contact</a>
            </div>
            <div className="flex items-center gap-4">
              <Link href="/" className="text-gray-300 hover:text-white transition-colors">
                Sign In
              </Link>
              <Link href="/" className="btn-primary text-sm py-2 px-4">
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 relative overflow-hidden">
        {/* Background effects */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-ona-primary/20 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-ona-secondary/20 rounded-full blur-3xl" />

        <div className="max-w-7xl mx-auto text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-6">
              <Heart className="w-4 h-4 text-red-400" />
              <span className="text-sm text-gray-300">Saving lives through early detection</span>
            </div>
            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6">
              AI-Powered<br />
              <span className="text-gradient">Medical Imaging</span>
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-8">
              Edge-first diagnostics for Africa. Fast, accurate, offline-capable screening
              for TB, Pneumonia, Cardiac conditions and more.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Link href="/" className="btn-primary flex items-center gap-2">
                Get Started <ArrowRight className="w-5 h-5" />
              </Link>
              <a href="#features" className="glass px-6 py-3 rounded-xl text-white hover:border-ona-primary/50 transition-all">
                Learn More
              </a>
            </div>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-20"
          >
            {stats.map((stat, index) => (
              <div key={stat.label} className="glass-card rounded-2xl p-6">
                <p className="text-3xl md:text-4xl font-bold text-gradient">{stat.value}</p>
                <p className="text-gray-400 mt-1">{stat.label}</p>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-4">
              Built for Healthcare Workers
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Every feature designed with clinical workflows in mind. Simple, fast, reliable.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="glass-card rounded-2xl p-6 hover:border-ona-primary/30 transition-all group"
              >
                <div className="w-12 h-12 rounded-xl bg-ona-primary/10 flex items-center justify-center mb-4 group-hover:bg-ona-primary/20 transition-colors">
                  <feature.icon className="w-6 h-6 text-ona-primary" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-gray-400">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-20 px-4 relative">
        <div className="absolute inset-0 bg-ona-glow opacity-30" />
        <div className="max-w-7xl mx-auto relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <h2 className="text-4xl font-bold text-white mb-6">
                Our Mission
              </h2>
              <p className="text-gray-400 mb-6">
                Millions die each year from treatable conditions — TB, pneumonia, heart disease —
                simply because their local clinic lacks a specialist to read a scan.
                Early detection saves lives, but access to expertise remains limited.
              </p>
              <p className="text-gray-400 mb-6">
                ONA Health brings AI-powered diagnostics to the point of care. Our edge-first approach
                means clinics can screen for 14+ conditions even without reliable internet, ensuring
                no one is left behind.
              </p>
              <div className="space-y-3">
                {[
                  'Screen for TB, pneumonia, cardiac conditions and more',
                  'Results in under 2 minutes, works offline',
                  'Multi-language support (English, Swahili, Somali)',
                  'Seamless integration with any X-ray equipment',
                ].map((item) => (
                  <div key={item} className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-ona-primary flex-shrink-0" />
                    <span className="text-gray-300">{item}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="glass-card rounded-3xl p-8"
            >
              <h3 className="text-2xl font-bold text-white mb-6">What We Detect</h3>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { condition: 'Tuberculosis', icon: '🫁' },
                  { condition: 'Pneumonia', icon: '🦠' },
                  { condition: 'Cardiomegaly', icon: '❤️' },
                  { condition: 'Pleural Effusion', icon: '💧' },
                  { condition: 'Atelectasis', icon: '🔬' },
                  { condition: 'Consolidation', icon: '📊' },
                ].map((item) => (
                  <div key={item.condition} className="flex items-center gap-3 p-3 rounded-xl bg-white/5">
                    <span className="text-xl">{item.icon}</span>
                    <span className="text-gray-300 text-sm">{item.condition}</span>
                  </div>
                ))}
              </div>
              <p className="text-gray-500 text-sm mt-4 text-center">
                + 8 more conditions powered by TorchXRayVision
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="glass-card rounded-3xl p-8 md:p-12"
          >
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
              <div>
                <h2 className="text-4xl font-bold text-white mb-4">Get in Touch</h2>
                <p className="text-gray-400 mb-8">
                  Interested in bringing ONA Health to your facility? Let's talk about how we can help.
                </p>
                <div className="space-y-4">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-ona-primary/10 flex items-center justify-center">
                      <Mail className="w-5 h-5 text-ona-primary" />
                    </div>
                    <div>
                      <p className="text-gray-400 text-sm">Email</p>
                      <p className="text-white">contact@onahealth.ai</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-ona-primary/10 flex items-center justify-center">
                      <Globe className="w-5 h-5 text-ona-primary" />
                    </div>
                    <div>
                      <p className="text-gray-400 text-sm">Website</p>
                      <p className="text-white">onahealth.ai</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-ona-primary/10 flex items-center justify-center">
                      <MapPin className="w-5 h-5 text-ona-primary" />
                    </div>
                    <div>
                      <p className="text-gray-400 text-sm">Location</p>
                      <p className="text-white">Nairobi, Kenya</p>
                    </div>
                  </div>
                </div>
              </div>

              <form className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <input type="text" placeholder="Your Name" className="input-glass" />
                  <input type="email" placeholder="Email Address" className="input-glass" />
                </div>
                <input type="text" placeholder="Organization" className="input-glass" />
                <textarea placeholder="Your Message" rows={4} className="input-glass resize-none" />
                <button type="submit" className="btn-primary w-full">
                  Send Message
                </button>
              </form>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 border-t border-white/5">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-ona-primary to-ona-secondary flex items-center justify-center">
              <Activity className="w-4 h-4 text-white" />
            </div>
            <span className="text-gray-400">2026 ONA Health. All rights reserved.</span>
          </div>
          <p className="text-ona-primary font-medium">See Clearly. Act Quickly.</p>
        </div>
      </footer>
    </div>
  )
}
