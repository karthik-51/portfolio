import { Card } from "@/components/ui/card";
import { Github, Linkedin, Mail } from "lucide-react";
import { HoverBorderGradient } from "@/components/ui/hover-border-gradient";

export function ContactSection() {
  return (
    <section className="py-12">
      <Card className="p-6 bg-card/50 backdrop-blur-sm border-primary/10">
        <h2 className="text-3xl font-semibold mb-6 text-center">
          Let&apos;s Connect
        </h2>
        <div className="flex flex-wrap justify-center gap-4">
          <HoverBorderGradient
            containerClassName="rounded-full"
            as="button"
            className="text-center dark:bg-sidebar-accent bg-white text-black dark:text-white flex items-center"
          >
            <a
              href="mailto:mandhalakarthikreddy@gmail.com"
              className="flex items-center gap-2 px-4 py-2"
            >
              <Mail className="h-5 w-5" />
              Email
            </a>
          </HoverBorderGradient>

          {/* LinkedIn Button */}
          <HoverBorderGradient
            containerClassName="rounded-full"
            as="button"
            className="text-center dark:bg-sidebar-accent bg-white text-black dark:text-white flex items-center"
          >
            <a
              href="https://www.linkedin.com/in/karthik-reddy-mandhala-57820925b/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2"
            >
              <Linkedin className="h-5 w-5" />
              LinkedIn
            </a>
          </HoverBorderGradient>

          {/* GitHub Button */}
          <HoverBorderGradient
            containerClassName="rounded-full"
            as="button"
            className="text-center dark:bg-sidebar-accent bg-white text-black dark:text-white flex items-center"
          >
            <a
              href="https://github.com/karthik-51"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2"
            >
              <Github className="h-5 w-5" />
              GitHub
            </a>
          </HoverBorderGradient>
        </div>
      </Card>
    </section>
  );
}