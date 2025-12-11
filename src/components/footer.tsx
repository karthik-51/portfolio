import { Button } from "@/components/ui/button"
import { Github, Linkedin, Mail } from "lucide-react"
import { IconEggCracked } from "@tabler/icons-react";
import Link from "next/link";

export function Footer() {
  return (
    <footer className="bg-card/30 backdrop-blur-sm border-t border-primary/10 mt-16 w-dvw">
      <div className="container mx-auto px-4 lg:px-0 max-w-4xl py-8">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="text-center md:text-left">
            <span className="text-muted-foreground">
              2025 - Karthik Reddy
            </span>
          </div>

          <div className="flex flex-col sm:flex-row items-center gap-4">
            <div className="flex gap-2">
              <Button variant="ghost" size="icon" asChild>
                <a href="mailto:mandhalakarthikreddy@gmail.com" aria-label="Email">
                  <Mail className="w-8 h-8" />
                </a>
              </Button>
              <Button variant="ghost" size="icon" asChild>
                <Link
                  href="https://www.linkedin.com/in/karthik-reddy-mandhala-57820925b/"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="LinkedIn"
                >
                  <Linkedin className="w-8 h-8" />
                </Link>
              </Button>
              <Button variant="ghost" size="icon" asChild>
                <Link
                  href="https://github.com/karthik-51"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="GitHub"
                >
                  <Github className="w-8 h-8" />
                </Link>
              </Button>
              <Button variant="ghost" size="icon" asChild>
                <Link
                  href="https://binarypiano.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="easter egg"
                >
                  <IconEggCracked stroke={2} />
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}