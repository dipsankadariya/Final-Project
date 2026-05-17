import { ArrowRight, Scale, BookOpen, Users, Zap, Shield, BarChart3, Lock, Lightbulb, Briefcase, Eye, Sparkles, CheckCircle2 } from 'lucide-react'
import { Link } from 'react-router-dom'

export function Landing() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-white via-blue-50/30 to-white text-gray-900">
      {/* Navigation */}
      <nav className="border-b border-gray-100 sticky top-0 z-50 bg-white/70 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-purple-600 to-blue-600 flex items-center justify-center text-white shadow-lg">
              <Scale className="w-5 h-5" strokeWidth={2.5} />
            </div>
            <span className="font-black text-xl tracking-tight">न्याय</span>
          </div>
          <div className="flex items-center gap-8">
            <Link to="/" className="text-sm text-purple-600 font-semibold hover:text-purple-700 transition-colors">
              Home
            </Link>
            <Link to="/about" className="text-sm text-gray-600 hover:text-gray-900 font-medium transition-colors">
              About
            </Link>
            <Link to="/forum" className="text-sm text-gray-600 hover:text-gray-900 font-medium transition-colors">
              Forum
            </Link>
            <Link
              to="/chat"
              className="px-6 py-2.5 text-sm font-semibold text-white bg-purple-600 hover:bg-purple-700 rounded transition-all shadow-md hover:shadow-lg hover:shadow-purple-200/50 hover:-translate-y-0.5"
            >
              Ask Now
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden pt-12 pb-20 md:pt-20 md:pb-32">
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-br from-blue-100 to-transparent rounded-full blur-3xl opacity-40 -translate-y-1/2 translate-x-1/3" />
          <div className="absolute -bottom-20 left-0 w-72 h-72 bg-gradient-to-tr from-purple-100 to-transparent rounded-full blur-3xl opacity-30 translate-y-1/3" />
        </div>
        
        <div className="max-w-7xl mx-auto px-6 relative z-10">
          <div className="grid md:grid-cols-2 gap-16 items-center">
            <div>
              <div className="inline-flex items-center gap-2 mb-6 px-4 py-2 bg-blue-50 rounded-full border border-blue-100">
                <Sparkles className="w-4 h-4 text-blue-600" />
                <span className="text-xs font-semibold text-blue-700">AI-Powered Legal Guidance</span>
              </div>
              
              <h1 className="text-6xl md:text-7xl font-black leading-tight mb-6 tracking-tight bg-gradient-to-r from-gray-950 via-gray-900 to-purple-900 bg-clip-text text-transparent">
                Your legal questions answered instantly
              </h1>
              
              <p className="text-lg text-gray-600 font-medium leading-relaxed mb-10 max-w-lg">
                Get reliable legal guidance grounded in Nepal's Constitution and laws. Clear answers, no legal jargon. Available 24/7.
              </p>

              {/* CTA Buttons */}
              <div className="flex flex-col sm:flex-row gap-4">
                <Link
                  to="/chat"
                  className="px-8 py-4 text-base font-semibold text-white bg-purple-600 hover:bg-purple-700 rounded transition-all shadow-lg shadow-purple-200/50 hover:shadow-xl hover:shadow-purple-300/50 hover:-translate-y-1 flex items-center justify-center gap-2"
                >
                  Ask Now <ArrowRight className="w-5 h-5" />
                </Link>
                <Link
                  to="/about"
                  className="px-8 py-4 text-base font-semibold text-gray-900 bg-white border-2 border-gray-200 hover:border-gray-300 hover:bg-gray-50 rounded transition-all"
                >
                  Learn More
                </Link>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-3 gap-6 mt-12 pt-12 border-t border-gray-200">
                <div>
                  <div className="text-3xl font-black text-gray-950 mb-1">2076</div>
                  <div className="text-sm text-gray-600 font-medium">Constitutional Articles</div>
                </div>
                <div>
                  <div className="text-3xl font-black text-gray-950 mb-1">50+</div>
                  <div className="text-sm text-gray-600 font-medium">Acts & Laws</div>
                </div>
                <div>
                  <div className="text-3xl font-black text-gray-950 mb-1">24/7</div>
                  <div className="text-sm text-gray-600 font-medium">Always Available</div>
                </div>
              </div>
            </div>

            {/* Hero Visual - Justice Theme */}
            <div className="relative h-96 md:h-full min-h-96">
              {/* Center - Scale Icon Only */}
              <div className="absolute inset-0 flex items-start justify-center pr-12 pt-16">
                <div className="relative flex flex-col items-center justify-center">
                  {/* Just the scale icon */}
                  <div className="flex items-center justify-center mb-6">
                    <Scale className="w-48 h-48 text-purple-600" strokeWidth={0.5} />
                  </div>
                  
                  {/* Text below logo */}
                  <div className="text-center">
                    <p className="text-sm font-semibold text-purple-700 tracking-wide">Justice • Equality • Trust</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Core Features */}
      <section className="max-w-7xl mx-auto px-6 py-20">
        <div className="mb-16">
          <h2 className="text-5xl font-black mb-4 tracking-tight text-gray-950">Why choose न्याय</h2>
          <p className="text-lg text-gray-600 font-medium max-w-2xl">Get instant access to Nepal's legal information with simple, accurate answers you can trust.</p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-8">
          <div className="group p-8 rounded-2xl bg-gradient-to-br from-blue-50 to-blue-50/30 border border-blue-100 hover:border-blue-200 transition-all hover:shadow-xl hover:shadow-blue-100/50">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-blue-100 to-blue-50 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Zap className="w-7 h-7 text-blue-600" strokeWidth={2} />
            </div>
            <h3 className="text-2xl font-black mb-3 text-gray-950">Instant Answers</h3>
            <p className="text-gray-700 font-medium leading-relaxed">Get legal answers in seconds. No complex documents, no legal jargon.</p>
          </div>
          
          <div className="group p-8 rounded-2xl bg-gradient-to-br from-purple-50 to-purple-50/30 border border-purple-100 hover:border-purple-200 transition-all hover:shadow-xl hover:shadow-purple-100/50">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-purple-100 to-purple-50 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <BookOpen className="w-7 h-7 text-purple-600" strokeWidth={2} />
            </div>
            <h3 className="text-2xl font-black mb-3 text-gray-950">Verified Sources</h3>
            <p className="text-gray-700 font-medium leading-relaxed">Every answer is backed by Nepal's Constitution, Acts, and official regulations.</p>
          </div>
          
          <div className="group p-8 rounded-2xl bg-gradient-to-br from-emerald-50 to-emerald-50/30 border border-emerald-100 hover:border-emerald-200 transition-all hover:shadow-xl hover:shadow-emerald-100/50">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-emerald-100 to-emerald-50 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <CheckCircle2 className="w-7 h-7 text-emerald-600" strokeWidth={2} />
            </div>
            <h3 className="text-2xl font-black mb-3 text-gray-950">Accurate & Reliable</h3>
            <p className="text-gray-700 font-medium leading-relaxed">Get consistent, accurate guidance based on official legal documents.</p>
          </div>
        </div>
      </section>
      {/* How It Works */}
      <section className="max-w-7xl mx-auto px-6 py-20 border-t border-gray-100">
        <div className="mb-16">
          <h2 className="text-5xl font-black mb-4 tracking-tight text-gray-950">How it works</h2>
          <p className="text-lg text-gray-600 font-medium">Three simple steps to get your legal answers</p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-8">
          {[
            { 
              num: '01', 
              title: 'Ask Your Question', 
              desc: 'Type any legal question in Nepali or English. Any topic related to Nepal\'s laws.',
              icon: BookOpen,
              bgColor: 'bg-blue-100'
            },
            { 
              num: '02', 
              title: 'Get Instant Results', 
              desc: 'Our system finds the exact legal information you need in seconds.',
              icon: Zap,
              bgColor: 'bg-purple-100'
            },
            { 
              num: '03', 
              title: 'Clear Guidance', 
              desc: 'Get easy-to-understand answers with legal references you can verify.',
              icon: Shield,
              bgColor: 'bg-emerald-100'
            },
          ].map((step, i) => (
            <div key={i} className="relative group">
              <div className={`${step.bgColor} rounded-2xl w-16 h-16 flex items-center justify-center mb-8 group-hover:scale-110 transition-transform`}>
                <step.icon className="w-8 h-8 text-gray-900" strokeWidth={1.5} />
              </div>
              <div className="text-4xl font-black text-gray-950/20 mb-2">{step.num}</div>
              <h3 className="text-xl font-black mb-3 text-gray-950">{step.title}</h3>
              <p className="text-gray-600 text-base leading-relaxed font-medium">{step.desc}</p>
              {i < 2 && (
                <div className="hidden md:block absolute top-8 -right-6 w-12 h-px bg-gradient-to-r from-gray-300 to-transparent" />
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Why Nepal Needs This */}
      <section className="max-w-7xl mx-auto px-6 py-20 border-t border-gray-100">
        <div className="mb-16">
          <h2 className="text-5xl font-black mb-4 tracking-tight text-gray-950">The Problem We Solve</h2>
          <p className="text-lg text-gray-600 font-medium max-w-2xl">Access to legal information shouldn't be a privilege</p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-8">
          <div className="group p-8 rounded-2xl bg-gradient-to-br from-red-50 to-red-50/30 border border-red-100 hover:border-red-200 hover:shadow-xl hover:shadow-red-100/30 transition-all">
            <div className="w-12 h-12 rounded-xl bg-red-100 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Lock className="w-6 h-6 text-red-600" strokeWidth={2} />
            </div>
            <h3 className="text-xl font-black mb-3 text-gray-950">Limited Access</h3>
            <p className="text-gray-700 font-medium leading-relaxed">Most Nepali citizens don't have easy access to understand their legal rights and responsibilities.</p>
          </div>
          
          <div className="group p-8 rounded-2xl bg-gradient-to-br from-yellow-50 to-yellow-50/30 border border-yellow-100 hover:border-yellow-200 hover:shadow-xl hover:shadow-yellow-100/30 transition-all">
            <div className="w-12 h-12 rounded-xl bg-yellow-100 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Lightbulb className="w-6 h-6 text-yellow-600" strokeWidth={2} />
            </div>
            <h3 className="text-xl font-black mb-3 text-gray-950">Complex Language</h3>
            <p className="text-gray-700 font-medium leading-relaxed">Legal documents use complex terminology that ordinary people find difficult to understand.</p>
          </div>
          
          <div className="group p-8 rounded-2xl bg-gradient-to-br from-blue-50 to-blue-50/30 border border-blue-100 hover:border-blue-200 hover:shadow-xl hover:shadow-blue-100/30 transition-all">
            <div className="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Briefcase className="w-6 h-6 text-blue-600" strokeWidth={2} />
            </div>
            <h3 className="text-xl font-black mb-3 text-gray-950">Expensive Help</h3>
            <p className="text-gray-700 font-medium leading-relaxed">Professional legal consultation is costly, making it inaccessible to average citizens.</p>
          </div>
          
          <div className="group p-8 rounded-2xl bg-gradient-to-br from-emerald-50 to-emerald-50/30 border border-emerald-100 hover:border-emerald-200 hover:shadow-xl hover:shadow-emerald-100/30 transition-all">
            <div className="w-12 h-12 rounded-xl bg-emerald-100 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Eye className="w-6 h-6 text-emerald-600" strokeWidth={2} />
            </div>
            <h3 className="text-xl font-black mb-3 text-gray-950">Trusted Information</h3>
            <p className="text-gray-700 font-medium leading-relaxed">न्याय provides sources so you can verify every answer against official documents.</p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="max-w-7xl mx-auto px-6 py-24 border-t border-gray-100 text-center">
        <div className="max-w-3xl mx-auto">
          <h2 className="text-6xl font-black mb-6 tracking-tight text-gray-950">Justice for everyone</h2>
          <p className="text-xl text-gray-600 font-medium leading-relaxed mb-12">
            Get the legal clarity you need. Understand your rights. Make informed decisions. All free, all in Nepali.
          </p>
          <Link
            to="/chat"
            className="inline-flex px-10 py-5 text-lg font-semibold text-white bg-purple-600 hover:bg-purple-700 rounded transition-all shadow-xl shadow-purple-200/50 hover:shadow-2xl hover:shadow-purple-300/50 hover:-translate-y-1 gap-2 items-center"
          >
            Start Asking <ArrowRight className="w-6 h-6" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-100 py-16 bg-gradient-to-b from-white to-gray-50/50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center">
            <p className="text-gray-600 font-medium">Built for Nepal. Making legal information accessible to everyone.</p>
            <p className="text-gray-500 text-sm mt-4">न्याय - Your guide to Nepal's legal system</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
