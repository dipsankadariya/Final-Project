import { Link } from 'react-router-dom'
import { Scale, BookOpen, Users, Zap } from 'lucide-react'

export function About() {
  return (
    <div className="min-h-screen bg-white text-gray-900">
      {/* Navigation */}
      <nav className="border-b border-gray-200 sticky top-0 z-50 bg-white/95 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-6 py-5 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-md bg-purple-600 flex items-center justify-center text-white">
              <Scale className="w-5 h-5" />
            </div>
            <span className="font-semibold text-lg tracking-tight">न्याय</span>
          </Link>
          <div className="flex items-center gap-8">
            <Link to="/" className="text-sm text-gray-600 hover:text-gray-900 transition-colors">
              Home
            </Link>
            <Link to="/about" className="text-sm text-purple-600 font-medium">
              About
            </Link>
            <Link to="/chat" className="px-5 py-2.5 text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 rounded-md transition-colors">
              Get Started
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-6 py-20">
        <h1 className="text-5xl font-bold mb-6 tracking-tight">About न्याय</h1>
        <p className="text-xl text-gray-600 max-w-3xl leading-relaxed">
          न्याय is a Constitution-Centric RAG system built to make Nepal's legal information accessible to everyone. We combine cutting-edge AI with Nepal's legal framework to provide instant, accurate legal guidance.
        </p>
      </section>

      {/* Mission */}
      <section className="max-w-6xl mx-auto px-6 py-20 border-t border-gray-200">
        <h2 className="text-3xl font-bold mb-8 tracking-tight">Our Mission</h2>
        <div className="grid md:grid-cols-2 gap-16">
          <div>
            <p className="text-lg text-gray-600 leading-relaxed">
              We believe legal knowledge should be accessible to all citizens of Nepal, regardless of their background or resources. Our mission is to bridge the gap between complex legal documents and everyday people.
            </p>
          </div>
          <div>
            <p className="text-lg text-gray-600 leading-relaxed">
              By leveraging advanced language models and retrieval-augmented generation (RAG), we provide instant answers grounded in Nepal's Constitution, Acts, and Regulations.
            </p>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="max-w-6xl mx-auto px-6 py-20 border-t border-gray-200">
        <h2 className="text-3xl font-bold mb-12 tracking-tight">How It Works</h2>
        <div className="grid md:grid-cols-3 gap-8">
          {[
            {
              icon: BookOpen,
              title: 'Constitution-Backed',
              desc: 'All answers reference Nepal\'s Constitution, Acts, and legal frameworks.',
            },
            {
              icon: Zap,
              title: 'Instant Retrieval',
              desc: 'Advanced AI retrieves relevant legal sections in milliseconds.',
            },
            {
              icon: Users,
              title: 'Transparent',
              desc: 'Every answer cites specific articles and sections for verification.',
            },
          ].map((item, i) => (
            <div key={i} className="border border-gray-200 rounded-lg p-6 hover:border-purple-200 transition-colors">
              <item.icon className="w-8 h-8 text-purple-600 mb-4" />
              <h3 className="font-bold text-lg mb-2">{item.title}</h3>
              <p className="text-gray-600 text-sm">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Technology */}
      <section className="max-w-6xl mx-auto px-6 py-20 border-t border-gray-200">
        <h2 className="text-3xl font-bold mb-8 tracking-tight">Built With</h2>
        <div className="grid md:grid-cols-2 gap-12">
          <div>
            <h3 className="font-bold text-lg mb-4">AI & Machine Learning</h3>
            <ul className="space-y-2 text-gray-600 text-sm">
              <li>• Fine-tuned Qwen 2.5 for Nepali legal language understanding</li>
              <li>• Hypothetical Document Embeddings (HyDE) for enhanced retrieval</li>
              <li>• LaBSE embeddings for multilingual support</li>
              <li>• FAISS for efficient vector similarity search</li>
            </ul>
          </div>
          <div>
            <h3 className="font-bold text-lg mb-4">Infrastructure</h3>
            <ul className="space-y-2 text-gray-600 text-sm">
              <li>• FastAPI for high-performance backend</li>
              <li>• React + Vite for modern frontend</li>
              <li>• Google OAuth for secure authentication</li>
              <li>• RAG architecture for reliable answers</li>
            </ul>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-6xl mx-auto px-6 py-20 border-t border-gray-200 text-center">
        <h2 className="text-3xl font-bold mb-4 tracking-tight">Ready to Get Started?</h2>
        <p className="text-gray-600 mb-8">Ask any legal question and get instant guidance grounded in Nepal's laws.</p>
        <Link
          to="/chat"
          className="inline-block px-6 py-3 text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 rounded-md transition-colors"
        >
          Start for Free
        </Link>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-200 py-12 bg-gray-50">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center text-sm text-gray-500">
            <p>Built for Nepal. Powered by Constitution-Centric RAG.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
